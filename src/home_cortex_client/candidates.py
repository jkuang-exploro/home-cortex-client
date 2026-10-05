"""Turn visual-change boundaries into local candidate clips.

The engine records ``upload_state=local_only`` and does not open a connection.
A later publisher may mark ``pending`` or ``transferred``. The clip bytes stay
in this store. They reuse the Stage 1 manifest and ``HCCLIP1`` container. A
motion interval shorter than ``merge_gap_s`` after the previous one extends
that candidate.
An interval marked ``forced_split`` is packaged on its own so a long movement
becomes several bounded clips instead of one long file. A scene change inside
an open motion interval stays on that motion candidate.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

from .buffer import BufferError, RingBuffer
from .config import ChangeDetectionConfig
from .detect import MotionEvent, SceneChangeEvent
from .evidence import (
    VISUAL_CHANGE,
    EvidenceFailure,
    build_manifest,
    encode_clip,
    evidence_id_for,
)


CANDIDATE_IDENTITY = (
    "embodiment_id",
    "camera_id",
    "trigger_type",
    "trigger_started_at",
    "trigger_ended_at",
    "clip_start",
    "clip_end",
    "peak_score",
    "evidence_id",
)
LOCAL = "local"
REQUESTED = "requested"
TRANSFERRED = "transferred"
EXPIRED = "expired"
LOCAL_ONLY = "local_only"
_CANDIDATE_KEY = re.compile(r"[0-9a-f]{32}\Z")


class CandidateFailure(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def candidate_id_for(fields: dict[str, object]) -> str:
    body = {key: fields[key] for key in CANDIDATE_IDENTITY}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "candidate:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


@dataclass
class _Pending:
    trigger_type: str
    started: datetime
    ended: datetime
    peak: float
    average: float
    samples: int
    scene_peak: float | None
    force_terminal: bool
    detector: str


class CandidateStore:
    """Bounded candidate metadata and clip bytes. Expired clips lose their bytes."""

    def __init__(
        self,
        root: Path,
        *,
        max_items: int,
        max_age_s: float,
        max_bytes: int,
    ) -> None:
        self.root = Path(root)
        self.max_items = max_items
        self.max_age_s = max_age_s
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict[str, object]] = {}
        self._stored: dict[str, datetime] = {}
        self._expired_at: dict[str, datetime] = {}
        self._lock = Lock()
        self.expired_total = 0
        self.expired_by_age = 0
        self.expired_by_count = 0
        self.expired_by_bytes = 0
        self.bytes_written_total = 0
        self._load_existing()

    def save(
        self,
        record: dict[str, object],
        manifest: dict[str, object],
        payload: bytes,
        *,
        now: datetime,
    ) -> dict[str, object]:
        candidate_id = str(record["candidate_id"])
        directory = self._directory(candidate_id)
        with self._lock:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "payload").write_bytes(payload)
            self.bytes_written_total += len(payload)
            (directory / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            self._records[candidate_id] = dict(record)
            self._stored[candidate_id] = now
            self._expired_at.pop(candidate_id, None)
            self._write_record(candidate_id)
            self._enforce(now)
            return dict(self._records[candidate_id])

    def get(self, candidate_id: str) -> dict[str, object]:
        with self._lock:
            record = self._records.get(candidate_id)
        if record is None:
            raise CandidateFailure("not_found", "candidate was not found")
        return dict(record)

    def inspect(self, candidate_id: str) -> dict[str, object]:
        record = self.get(candidate_id)
        provenance = {
            "embodiment_id": record["embodiment_id"],
            "camera_id": record["camera_id"],
            "detector": record["detector"],
            "detector_config": record.get("detector_config"),
            "raw_event_interval": {
                "start": record["trigger_started_at"],
                "end": record["trigger_ended_at"],
            },
            "requested_interval": {
                "start": record.get("requested_start"),
                "end": record.get("requested_end"),
            },
            "final_evidence_interval": {
                "start": record["clip_start"],
                "end": record["clip_end"],
            },
            "evidence_id": record["evidence_id"],
            "sequence_range": {
                "start": record.get("sequence_start"),
                "end": record.get("sequence_end"),
            },
            "media_sha256": record.get("media_sha256"),
            "transfer": {
                "state": record["upload_state"],
                "has_left_client": record["upload_state"] == "transferred",
                "requested_at": record.get("requested_at"),
                "transferred_at": record.get("transferred_at"),
            },
            "local_clip_path": (
                str(self._directory(candidate_id) / "payload")
                if record["status"] == LOCAL else None
            ),
        }
        if record["status"] != LOCAL:
            return {
                "candidate": record,
                "provenance": provenance,
                "manifest": None,
                "manifest_ok": False,
                "hash_ok": False,
                "byte_length": 0,
                "evidence_available": False,
            }
        payload = self._payload(candidate_id)
        manifest = self._manifest(candidate_id)
        digest = hashlib.sha256(payload).hexdigest()
        try:
            manifest_ok = (
                evidence_id_for(manifest) == manifest.get("evidence_id")
                and manifest.get("evidence_id") == record["evidence_id"]
                and manifest.get("embodiment_id") == record["embodiment_id"]
                and manifest.get("camera_id") == record["camera_id"]
                and manifest.get("reason") == VISUAL_CHANGE
            )
        except KeyError:
            manifest_ok = False
        return {
            "candidate": record,
            "provenance": provenance,
            "manifest": manifest,
            "manifest_ok": manifest_ok,
            "hash_ok": manifest_ok and digest == manifest.get("sha256"),
            "byte_length": len(payload),
            "evidence_available": True,
        }

    def set_upload_state(
        self,
        candidate_id: str,
        state: str,
        *,
        now: datetime,
        hold: str | None = None,
    ) -> dict[str, object]:
        """Record transfer progress. This does not delete clip bytes."""
        if state not in {LOCAL_ONLY, "pending", TRANSFERRED}:
            raise CandidateFailure("invalid_manifest", "upload state is invalid")
        if hold not in {None, "released", "rejected"}:
            raise CandidateFailure("invalid_manifest", "promotion queue hold is invalid")
        stamp = now.isoformat(timespec="milliseconds")
        with self._lock:
            record = self._records.get(candidate_id)
            if record is None:
                raise CandidateFailure("not_found", "candidate was not found")
            if record.get("status") != LOCAL:
                raise CandidateFailure("expired", "candidate clip is no longer available")
            record["upload_state"] = state
            if state == "pending":
                record["requested_at"] = stamp
                record.pop("promotion_queue", None)
            elif state == TRANSFERRED:
                record.setdefault("requested_at", stamp)
                record["transferred_at"] = stamp
                record.pop("promotion_queue", None)
            elif hold is not None:
                record["promotion_queue"] = hold
            self._write_record(candidate_id)
            return dict(record)

    def media(self, candidate_id: str) -> bytes:
        record = self.get(candidate_id)
        if record["status"] != LOCAL:
            raise CandidateFailure("expired", "candidate clip is no longer available")
        return self._payload(candidate_id)

    def records(self) -> list[dict[str, object]]:
        with self._lock:
            ordered = sorted(self._records, key=lambda candidate_id: self._stored[candidate_id])
            return [dict(self._records[candidate_id]) for candidate_id in ordered]

    def local_records(self) -> list[dict[str, object]]:
        with self._lock:
            ordered = sorted(
                (candidate_id for candidate_id, record in self._records.items() if record["status"] == LOCAL),
                key=lambda candidate_id: self._stored[candidate_id],
            )
            return [dict(self._records[candidate_id]) for candidate_id in ordered]

    def disk_bytes(self) -> int:
        with self._lock:
            total = 0
            for candidate_id, record in self._records.items():
                if record["status"] != LOCAL:
                    continue
                path = self._directory(candidate_id) / "payload"
                if path.is_file():
                    total += path.stat().st_size
            return total

    def _load_existing(self) -> None:
        """Rebuild the local index from retained candidate records on startup."""
        for directory in self.root.iterdir():
            if not directory.is_dir() or not _CANDIDATE_KEY.fullmatch(directory.name):
                continue
            path = directory / "candidate.json"
            if not path.is_file():
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                candidate_id = "candidate:" + directory.name
                if not isinstance(record, dict) or record.get("candidate_id") != candidate_id:
                    continue
                if record.get("status") not in {LOCAL, EXPIRED}:
                    continue
                stamp = record.get("retained_at")
                stored = (
                    datetime.fromisoformat(stamp)
                    if isinstance(stamp, str) else
                    datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                )
                if stored.tzinfo is None:
                    continue
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            self._records[candidate_id] = record
            self._stored[candidate_id] = stored
            if record["status"] == EXPIRED:
                self._expired_at[candidate_id] = stored
                self.expired_total += 1
        self._enforce(datetime.now(timezone.utc))

    def _payload(self, candidate_id: str) -> bytes:
        path = self._directory(candidate_id) / "payload"
        if not path.is_file():
            raise CandidateFailure("expired", "candidate clip is no longer available")
        return path.read_bytes()

    def _manifest(self, candidate_id: str) -> dict[str, object]:
        path = self._directory(candidate_id) / "manifest.json"
        if not path.is_file():
            raise CandidateFailure("expired", "candidate clip is no longer available")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise CandidateFailure("expired", "candidate clip is no longer available")
        return manifest

    def _enforce(self, now: datetime) -> None:
        for candidate_id, stored in list(self._stored.items()):
            record = self._records.get(candidate_id)
            if record is None:
                continue
            age = (now - stored).total_seconds()
            if record["status"] == LOCAL and age > self.max_age_s:
                self._expire(candidate_id, now, reason="age")
            elif record["status"] == EXPIRED:
                expired_at = self._expired_at.get(candidate_id, stored)
                if (now - expired_at).total_seconds() > self.max_age_s:
                    self._delete(candidate_id)
        locals_ = [cid for cid, record in self._records.items() if record["status"] == LOCAL]
        locals_.sort(key=lambda cid: self._stored[cid])
        overflow = len(locals_) - self.max_items
        for candidate_id in locals_[: max(overflow, 0)]:
            self._expire(candidate_id, now, reason="count")
        while self._disk_locked() > self.max_bytes:
            remaining = [cid for cid, record in self._records.items() if record["status"] == LOCAL]
            remaining.sort(key=lambda cid: self._stored[cid])
            if not remaining:
                return
            self._expire(remaining[0], now, reason="bytes")

    def _expire(self, candidate_id: str, now: datetime, *, reason: str) -> None:
        record = self._records[candidate_id]
        record["status"] = EXPIRED
        self._expired_at[candidate_id] = now
        self.expired_total += 1
        if reason == "age":
            self.expired_by_age += 1
        elif reason == "count":
            self.expired_by_count += 1
        else:
            self.expired_by_bytes += 1
        directory = self._directory(candidate_id)
        for name in ("payload", "manifest.json"):
            path = directory / name
            if path.is_file():
                path.unlink()
        self._write_record(candidate_id)

    def _delete(self, candidate_id: str) -> None:
        self._records.pop(candidate_id, None)
        self._stored.pop(candidate_id, None)
        self._expired_at.pop(candidate_id, None)
        directory = self._directory(candidate_id)
        if directory.exists():
            for child in directory.iterdir():
                child.unlink()
            directory.rmdir()

    def _disk_locked(self) -> int:
        total = 0
        for candidate_id, record in self._records.items():
            if record["status"] != LOCAL:
                continue
            path = self._directory(candidate_id) / "payload"
            if path.is_file():
                total += path.stat().st_size
        return total

    def _write_record(self, candidate_id: str) -> None:
        directory = self._directory(candidate_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "candidate.json").write_text(
            json.dumps(self._records[candidate_id], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _directory(self, candidate_id: str) -> Path:
        key = candidate_id.removeprefix("candidate:")
        if not _CANDIDATE_KEY.fullmatch(key) or candidate_id != "candidate:" + key:
            raise CandidateFailure("not_found", "candidate was not found")
        return self.root / key


class CandidateEngine:
    """Merge detector intervals and package one local clip after post-roll."""

    def __init__(
        self,
        buffer: RingBuffer,
        *,
        embodiment_id: str,
        camera_id: str,
        root: Path,
        config: ChangeDetectionConfig | None = None,
    ) -> None:
        self.buffer = buffer
        self.embodiment_id = embodiment_id
        self.camera_id = camera_id
        self.config = config or ChangeDetectionConfig()
        self.store = CandidateStore(
            root,
            max_items=self.config.max_candidates,
            max_age_s=self.config.max_candidate_age_s,
            max_bytes=self.config.max_candidate_disk_bytes,
        )
        self.merged_extensions = 0
        self.discarded = 0
        self.suppressed_duplicates = 0
        self.suppressed_cooldown = 0
        self.missed_buffer = 0
        self.folded_scenes = 0
        self.produced = len(self.store.records())
        self._open: _Pending | None = None
        self._sealing: list[_Pending] = []
        self._scenes: list[SceneChangeEvent] = []
        self._suppress_until: datetime | None = None
        self._last_finalized_at: datetime | None = None
        self._live_start: datetime | None = None
        self._live_end: datetime | None = None

    def note_motion(self, started: datetime | None, last_active: datetime | None) -> None:
        """Remember the detector interval that has not closed yet."""
        self._live_start = started
        self._live_end = last_active

    def ingest(self, signal: MotionEvent | SceneChangeEvent) -> None:
        if isinstance(signal, SceneChangeEvent):
            self._scenes.append(signal)
            return
        if self._open is not None and self._can_merge(self._open, signal):
            self._extend(self._open, signal)
            self.merged_extensions += 1
            return
        if self._open is not None:
            self._sealing.append(self._open)
            self._open = None
        if self._in_cooldown(signal.started_at):
            self.suppressed_cooldown += 1
            return
        self._open = _Pending(
            trigger_type="motion",
            started=signal.started_at,
            ended=signal.ended_at,
            peak=signal.peak_score,
            average=signal.average_score,
            samples=signal.sample_count,
            scene_peak=None,
            force_terminal=signal.forced_split,
            detector=signal.detector,
        )

    def advance(self, now: datetime) -> list[dict[str, object]]:
        if now.tzinfo is None:
            raise CandidateFailure("capture_failure", "candidate clock must be timezone-aware")
        packaged = self._finalize_ready(now)
        remaining: list[SceneChangeEvent] = []
        for scene in self._scenes:
            if self._folded(scene):
                self.folded_scenes += 1
                continue
            if self._inside_live_motion(scene.occurred_at):
                remaining.append(scene)
                continue
            if now < scene.occurred_at + timedelta(seconds=self.config.post_roll_s):
                remaining.append(scene)
                continue
            if not self.config.scene_change_candidates:
                continue
            if self._in_cooldown(scene.occurred_at):
                self.suppressed_cooldown += 1
                continue
            record = self._package(self._scene_pending(scene), now)
            if record is not None:
                packaged.append(record)
        self._scenes = remaining
        return packaged

    def stats(self) -> dict[str, object]:
        local = self.store.local_records()
        retained = 0.0
        longest = 0.0
        for record in local:
            duration = _trigger_duration(record)
            retained += duration
            longest = max(longest, duration)
        return {
            "candidates": len(local),
            "candidates_produced": self.produced,
            "candidate_bytes_written": self.store.bytes_written_total,
            "old_candidates_expired": self.store.expired_total,
            "expired_by_age": self.store.expired_by_age,
            "expired_by_count": self.store.expired_by_count,
            "expired_by_bytes": self.store.expired_by_bytes,
            "motion_candidates": sum(record["trigger_type"] == "motion" for record in local),
            "scene_candidates": sum(record["trigger_type"] == "scene_change" for record in local),
            "merged_extensions": self.merged_extensions,
            "discarded": self.discarded,
            "suppressed_duplicates": self.suppressed_duplicates,
            "suppressed_cooldown": self.suppressed_cooldown,
            "missed_buffer": self.missed_buffer,
            "folded_scenes": self.folded_scenes,
            "retained_trigger_s": round(retained, 3),
            "longest_trigger_s": round(longest, 3),
            "disk_bytes": self.store.disk_bytes(),
            "upload_states": sorted({str(record["upload_state"]) for record in local}),
        }

    def _finalize_ready(self, now: datetime) -> list[dict[str, object]]:
        ready: list[_Pending] = []
        waiting: list[_Pending] = []
        for pending in self._sealing:
            if self._ready(pending, now):
                ready.append(pending)
            else:
                waiting.append(pending)
        self._sealing = waiting
        if self._open is not None and self._ready(self._open, now):
            ready.append(self._open)
            self._open = None
        packaged: list[dict[str, object]] = []
        for pending in ready:
            record = self._package(pending, now)
            if record is not None:
                packaged.append(record)
        return packaged

    def _ready(self, pending: _Pending, now: datetime) -> bool:
        wait = self.config.post_roll_s
        if pending.trigger_type == "motion" and not pending.force_terminal:
            wait = max(wait, self.config.merge_gap_s)
            # The next interval is visible only after it closes. Hold this one
            # while the detector is already inside a gap that can still merge.
            if self._could_extend(pending):
                return False
        return now >= pending.ended + timedelta(seconds=wait)

    def _could_extend(self, pending: _Pending) -> bool:
        if self._live_start is None:
            return False
        gap = (self._live_start - pending.ended).total_seconds()
        return gap < self.config.merge_gap_s

    def _can_merge(self, pending: _Pending, event: MotionEvent) -> bool:
        if pending.force_terminal or pending.trigger_type != "motion":
            return False
        gap = (event.started_at - pending.ended).total_seconds()
        return gap < self.config.merge_gap_s

    def _extend(self, pending: _Pending, event: MotionEvent) -> None:
        weight = pending.samples + event.sample_count
        if weight:
            pending.average = (
                pending.average * pending.samples + event.average_score * event.sample_count
            ) / weight
        pending.samples = weight
        pending.peak = max(pending.peak, event.peak_score)
        if event.started_at < pending.started:
            pending.started = event.started_at
        if event.ended_at > pending.ended:
            pending.ended = event.ended_at
        if event.forced_split:
            pending.force_terminal = True

    def _folded(self, scene: SceneChangeEvent) -> bool:
        when = scene.occurred_at
        holders = list(self._sealing)
        if self._open is not None:
            holders.append(self._open)
        for pending in holders:
            if pending.trigger_type == "motion" and pending.started <= when <= pending.ended:
                pending.scene_peak = max(scene.score, pending.scene_peak or 0.0)
                return True
        for record in self.store.local_records():
            if record["trigger_type"] != "motion":
                continue
            start = datetime.fromisoformat(str(record["trigger_started_at"]))
            end = datetime.fromisoformat(str(record["trigger_ended_at"]))
            if start <= when <= end:
                return True
        return False

    def _scene_pending(self, scene: SceneChangeEvent) -> _Pending:
        return _Pending(
            trigger_type="scene_change",
            started=scene.occurred_at,
            ended=scene.occurred_at,
            peak=scene.score,
            average=scene.score,
            samples=1,
            scene_peak=scene.score,
            force_terminal=True,
            detector=scene.detector,
        )

    def _inside_live_motion(self, when: datetime) -> bool:
        if self._live_start is None or self._live_end is None:
            return False
        return self._live_start <= when <= self._live_end

    def _in_cooldown(self, when: datetime) -> bool:
        return (
            self._suppress_until is not None
            and self._last_finalized_at is not None
            and self._last_finalized_at <= when < self._suppress_until
        )

    def _package(self, pending: _Pending, now: datetime) -> dict[str, object] | None:
        duration = (pending.ended - pending.started).total_seconds()
        if pending.trigger_type == "motion" and (
            duration < self.config.minimum_event_duration_s
            or pending.peak < self.config.minimum_peak_score
        ):
            self.discarded += 1
            return None
        if pending.trigger_type != "motion" and pending.peak < self.config.minimum_peak_score:
            self.discarded += 1
            return None
        window_start = pending.started - timedelta(seconds=self.config.pre_roll_s)
        window_end = pending.ended + timedelta(seconds=self.config.post_roll_s)
        try:
            chosen = tuple(
                segment for segment in self.buffer.overlapping(window_start, window_end)
                if segment.camera_id == self.camera_id
            )
        except BufferError as error:
            raise CandidateFailure(error.code, str(error)) from error
        if not chosen:
            self.missed_buffer += 1
            return None
        if len({(segment.width, segment.height) for segment in chosen}) != 1:
            self.missed_buffer += 1
            return None
        sequences = [segment.sequence_number for segment in chosen]
        if sequences != list(range(sequences[0], sequences[-1] + 1)):
            self.missed_buffer += 1
            return None
        try:
            payload = encode_clip(chosen)
        except EvidenceFailure as error:
            raise CandidateFailure(error.code, str(error)) from error
        clip_start = chosen[0].captured_at
        clip_end = chosen[-1].ends_at()
        manifest = build_manifest(
            embodiment_id=self.embodiment_id,
            camera_id=chosen[0].camera_id,
            media_type="video_clip",
            captured_start=_iso(clip_start),
            captured_end=_iso(clip_end),
            duration_ms=_duration_ms(clip_start, clip_end),
            sequence_start=sequences[0],
            sequence_end=sequences[-1],
            width=chosen[0].width,
            height=chosen[0].height,
            content_type="application/x-home-cortex-clip",
            sha256=hashlib.sha256(payload).hexdigest(),
            reason=VISUAL_CHANGE,
        )
        peak = round(pending.peak, 6)
        fields = {
            "embodiment_id": self.embodiment_id,
            "camera_id": self.camera_id,
            "trigger_type": pending.trigger_type,
            "trigger_started_at": _iso(pending.started),
            "trigger_ended_at": _iso(pending.ended),
            "clip_start": _iso(clip_start),
            "clip_end": _iso(clip_end),
            "peak_score": peak,
            "evidence_id": manifest["evidence_id"],
        }
        if self._duplicate(fields):
            self.suppressed_duplicates += 1
            return None
        record: dict[str, object] = {
            **fields,
            "candidate_id": candidate_id_for(fields),
            "retained_at": _iso(now),
            "requested_start": _iso(window_start),
            "requested_end": _iso(window_end),
            "detector_config": asdict(self.config),
            "sequence_start": sequences[0],
            "sequence_end": sequences[-1],
            "media_sha256": manifest["sha256"],
            "average_score": round(pending.average, 6),
            "scene_change_score": None if pending.scene_peak is None else round(pending.scene_peak, 6),
            "detector": pending.detector,
            "upload_state": LOCAL_ONLY,
            "status": LOCAL,
            "sample_count": pending.samples,
        }
        saved = self.store.save(record, manifest, payload, now=now)
        self.produced += 1
        self._last_finalized_at = now
        if self.config.cooldown_s > 0:
            self._suppress_until = now + timedelta(seconds=self.config.cooldown_s)
        return saved

    def _duplicate(self, fields: dict[str, object]) -> bool:
        start = datetime.fromisoformat(str(fields["clip_start"]))
        end = datetime.fromisoformat(str(fields["clip_end"]))
        for record in self.store.local_records():
            if record["camera_id"] != fields["camera_id"]:
                continue
            if record["trigger_type"] != fields["trigger_type"]:
                continue
            other_start = datetime.fromisoformat(str(record["clip_start"]))
            other_end = datetime.fromisoformat(str(record["clip_end"]))
            if _overlap_fraction(start, end, other_start, other_end) >= 0.8:
                return True
        return False


def _trigger_duration(record: dict[str, object]) -> float:
    start = datetime.fromisoformat(str(record["trigger_started_at"]))
    end = datetime.fromisoformat(str(record["trigger_ended_at"]))
    return max(0.0, (end - start).total_seconds())


def _overlap_fraction(start: datetime, end: datetime, other_start: datetime, other_end: datetime) -> float:
    overlap = min(end, other_end) - max(start, other_start)
    if overlap.total_seconds() <= 0:
        return 0.0
    shorter = min(end - start, other_end - other_start).total_seconds()
    if shorter <= 0:
        return 1.0
    return overlap.total_seconds() / shorter


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds")


def _duration_ms(start: datetime, end: datetime) -> int:
    return int(round((end - start).total_seconds() * 1000))
