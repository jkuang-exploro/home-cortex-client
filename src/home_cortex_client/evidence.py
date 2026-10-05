"""Package selected camera bytes as VisualEvidence with a stable manifest.

The evidence id is a hash of the manifest fields. It is not a filename.
Home Cortex repeats this canonical JSON when it checks the id:

    json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Callable
import struct

from .buffer import BufferError, CapturedSegment, RingBuffer


IDENTITY_FIELDS = (
    "embodiment_id",
    "camera_id",
    "media_type",
    "captured_start",
    "captured_end",
    "duration_ms",
    "sequence_start",
    "sequence_end",
    "width",
    "height",
    "content_type",
    "sha256",
    "reason",
)
IMAGE_CONTENT_TYPE = "image/jpeg"
CLIP_CONTENT_TYPE = "application/x-home-cortex-clip"
CLIP_MAGIC = b"HCCLIP1"
MANUAL_OBSERVE = "manual_observe"
MANUAL_OBSERVE_CLIP = "manual_observe_clip"
VISUAL_CHANGE = "visual_change"


class EvidenceFailure(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class PackagedEvidence:
    manifest: dict[str, object]
    payload: bytes


def evidence_id_for(fields: dict[str, object]) -> str:
    body = {key: fields[key] for key in IDENTITY_FIELDS}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "evidence:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def build_manifest(**fields: object) -> dict[str, object]:
    manifest = {key: fields[key] for key in IDENTITY_FIELDS}
    manifest["evidence_id"] = evidence_id_for(manifest)
    return manifest


def encode_clip(segments: tuple[CapturedSegment, ...] | list[CapturedSegment]) -> bytes:
    """Length-prefixed JPEG sequence. Timing lives in the manifest, not a filename."""
    if not segments:
        raise EvidenceFailure("capture_failure", "clip contains no frames")
    parts = [CLIP_MAGIC, struct.pack(">I", len(segments))]
    previous_at: datetime | None = None
    for segment in segments:
        if not segment.payload.startswith(b"\xff\xd8") or not segment.payload.endswith(b"\xff\xd9"):
            raise EvidenceFailure("capture_failure", "clip frame is not a JPEG")
        if previous_at is not None and segment.captured_at < previous_at:
            raise EvidenceFailure("capture_failure", "clip frames are out of time order")
        previous_at = segment.captured_at
        millis = _unix_millis(segment.captured_at)
        parts.append(struct.pack(">QqI", segment.sequence_number, millis, len(segment.payload)))
        parts.append(segment.payload)
    return b"".join(parts)


def decode_clip(payload: bytes) -> list[tuple[int, int, bytes]]:
    if len(payload) < 11 or payload[:7] != CLIP_MAGIC:
        raise EvidenceFailure("invalid_manifest", "clip container is invalid")
    (count,) = struct.unpack_from(">I", payload, 7)
    if count < 1:
        raise EvidenceFailure("invalid_manifest", "clip container is invalid")
    offset = 11
    frames: list[tuple[int, int, bytes]] = []
    for _ in range(count):
        if offset + 20 > len(payload):
            raise EvidenceFailure("invalid_manifest", "clip container is truncated")
        sequence, millis, length = struct.unpack_from(">QqI", payload, offset)
        offset += 20
        if length < 1 or offset + length > len(payload):
            raise EvidenceFailure("invalid_manifest", "clip container is truncated")
        frame = payload[offset:offset + length]
        offset += length
        if not frame.startswith(b"\xff\xd8") or not frame.endswith(b"\xff\xd9"):
            raise EvidenceFailure("invalid_manifest", "clip frame is not a JPEG")
        frames.append((sequence, millis, frame))
    if offset != len(payload):
        raise EvidenceFailure("invalid_manifest", "clip container has trailing bytes")
    return frames


class LocalEvidenceStore:
    """Temporary selected evidence. Retention is by age and item count."""

    def __init__(self, root: Path, *, max_items: int = 8, max_age_s: float = 3600.0) -> None:
        if max_items < 1:
            raise ValueError("evidence retention must keep at least one item")
        self.root = Path(root)
        self.max_items = max_items
        self.max_age_s = max_age_s
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def put(self, manifest: dict[str, object], payload: bytes, *, now: datetime) -> Path:
        evidence_id = manifest["evidence_id"]
        if not isinstance(evidence_id, str) or ":" not in evidence_id:
            raise EvidenceFailure("invalid_manifest", "evidence_id is invalid")
        directory = self.root / evidence_id.split(":", 1)[1]
        with self._lock:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "payload").write_bytes(payload)
            (directory / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (directory / "stored_at").write_text(now.isoformat(), encoding="utf-8")
            self._cleanup_locked(now)
        return directory

    def inspect(self, evidence_id: str) -> dict[str, object]:
        with self._lock:
            found: tuple[Path, dict[str, object]] | None = None
            if self.root.exists():
                for child in self.root.iterdir():
                    manifest_path = child / "manifest.json"
                    payload_path = child / "payload"
                    if not manifest_path.is_file() or not payload_path.is_file():
                        continue
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if isinstance(manifest, dict) and manifest.get("evidence_id") == evidence_id:
                        found = (child, manifest)
                        break
            if found is None:
                raise EvidenceFailure("not_found", "evidence was not found")
            directory, manifest = found
            payload = (directory / "payload").read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        return {
            "manifest": manifest,
            "hash_ok": digest == manifest.get("sha256"),
            "byte_length": len(payload),
            "directory": str(directory),
        }

    def cleanup(self, now: datetime) -> None:
        with self._lock:
            self._cleanup_locked(now)

    def _cleanup_locked(self, now: datetime) -> None:
        entries: list[tuple[datetime, Path]] = []
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            stamp = _stored_at(child)
            age = (now - stamp).total_seconds()
            if age > self.max_age_s:
                shutil.rmtree(child, ignore_errors=True)
                continue
            entries.append((stamp, child))
        entries.sort(key=lambda item: item[0])
        overflow = len(entries) - self.max_items
        for _, child in entries[: max(overflow, 0)]:
            shutil.rmtree(child, ignore_errors=True)


class EvidencePackager:
    """Turn an explicit selection into a manifest plus media bytes."""

    def __init__(
        self,
        buffer: RingBuffer,
        store: LocalEvidenceStore,
        *,
        embodiment_id: str,
        freshness_s: float = 2.0,
        clock: Callable[[], datetime] | None = None,
        refresh: Callable[[], None] | None = None,
    ) -> None:
        if not embodiment_id.startswith("embodiment:"):
            raise ValueError("embodiment_id must be an embodiment: record ID")
        if not math.isfinite(freshness_s) or freshness_s <= 0:
            raise ValueError("freshness window must be positive")
        self.buffer = buffer
        self.store = store
        self.embodiment_id = embodiment_id
        self.freshness_s = float(freshness_s)
        self.clock = clock or (lambda: datetime.now().astimezone())
        self.refresh = refresh

    def get_latest_still(self, *, now: datetime | None = None) -> PackagedEvidence:
        segment = self._fresh_latest(now)
        payload = segment.payload
        if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
            raise EvidenceFailure("capture_failure", "latest frame is not a JPEG")
        stamp = _iso(segment.captured_at)
        manifest = build_manifest(
            embodiment_id=self.embodiment_id,
            camera_id=segment.camera_id,
            media_type="image",
            captured_start=stamp,
            captured_end=stamp,
            duration_ms=0,
            sequence_start=segment.sequence_number,
            sequence_end=segment.sequence_number,
            width=segment.width,
            height=segment.height,
            content_type=IMAGE_CONTENT_TYPE,
            sha256=hashlib.sha256(payload).hexdigest(),
            reason=MANUAL_OBSERVE,
        )
        self.store.put(manifest, payload, now=now or self.clock())
        return PackagedEvidence(manifest, payload)

    def get_recent_clip(self, duration_seconds: float, *, now: datetime | None = None) -> PackagedEvidence:
        try:
            requested = float(duration_seconds)
        except (TypeError, ValueError) as error:
            raise EvidenceFailure(
                "invalid_duration", "duration must be a positive number of seconds",
            ) from error
        current = self._resolve_now(now)
        try:
            self._require_fresh(current)
        except EvidenceFailure:
            if now is not None or self.refresh is None:
                raise
            self.refresh()
            current = self.clock()
            self._require_fresh(current)
        try:
            chosen = self.buffer.select_recent(requested, now=current)
        except BufferError as error:
            raise EvidenceFailure(error.code, str(error)) from error
        if len({(segment.width, segment.height) for segment in chosen}) != 1:
            raise EvidenceFailure("capture_failure", "clip frames must share one resolution")
        sequences = [segment.sequence_number for segment in chosen]
        if sequences != list(range(sequences[0], sequences[-1] + 1)):
            raise EvidenceFailure("capture_failure", "clip sequence is not contiguous")
        payload = encode_clip(chosen)
        captured_start = chosen[0].captured_at
        captured_end = chosen[-1].ends_at()
        duration_ms = _duration_ms(captured_start, captured_end)
        if duration_ms <= 0:
            raise EvidenceFailure("capture_failure", "clip duration must be positive")
        manifest = build_manifest(
            embodiment_id=self.embodiment_id,
            camera_id=chosen[0].camera_id,
            media_type="video_clip",
            captured_start=_iso(captured_start),
            captured_end=_iso(captured_end),
            duration_ms=duration_ms,
            sequence_start=sequences[0],
            sequence_end=sequences[-1],
            width=chosen[0].width,
            height=chosen[0].height,
            content_type=CLIP_CONTENT_TYPE,
            sha256=hashlib.sha256(payload).hexdigest(),
            reason=MANUAL_OBSERVE_CLIP,
        )
        self.store.put(manifest, payload, now=current)
        return PackagedEvidence(manifest, payload)

    def _fresh_latest(self, now: datetime | None) -> CapturedSegment:
        current = self._resolve_now(now)
        try:
            return self._require_fresh(current)
        except EvidenceFailure as error:
            if now is not None or self.refresh is None or error.code not in {
                "evidence_stale", "camera_unavailable",
            }:
                raise
        self.refresh()
        return self._require_fresh(self.clock())

    def _require_fresh(self, now: datetime) -> CapturedSegment:
        latest = self.buffer.latest()
        if latest is None:
            raise EvidenceFailure("camera_unavailable", "no captured frame is available")
        age = (now - latest.captured_at).total_seconds()
        if age < -2:
            raise EvidenceFailure("capture_failure", "capture timestamp is ahead of the client clock")
        if age > self.freshness_s:
            raise EvidenceFailure(
                "evidence_stale",
                "latest frame is older than the freshness window",
            )
        return latest

    def _resolve_now(self, now: datetime | None) -> datetime:
        current = self.clock() if now is None else now
        if current.tzinfo is None:
            raise EvidenceFailure("capture_failure", "clock must be timezone-aware")
        return current


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise EvidenceFailure("capture_failure", "capture time must be timezone-aware")
    return value.isoformat(timespec="milliseconds")


def _unix_millis(value: datetime) -> int:
    return int(round(value.timestamp() * 1000))


def _duration_ms(start: datetime, end: datetime) -> int:
    return int(round((end - start).total_seconds() * 1000))


def _stored_at(directory: Path) -> datetime:
    try:
        stamp = datetime.fromisoformat((directory / "stored_at").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        stamp = datetime.fromtimestamp(directory.stat().st_mtime, timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp
