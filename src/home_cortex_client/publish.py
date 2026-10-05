"""Push selected visual evidence through the one embodiment evidence protocol.

Capture, buffering, and the promotion decision stay on the client. This module
sends only ``PROMOTE`` envelopes. ``RETAIN_LOCAL`` and ``DROP`` never enter
the queue. A disconnected Home Cortex leaves the queue bounded and retryable.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from .buffer import CapturedSegment
from .evidence import CLIP_CONTENT_TYPE, IMAGE_CONTENT_TYPE, build_manifest, encode_clip
from .policy import POLICY_VERSION, PolicyDecision, parse_policy_decision
from .semantic import (
    SCHEMA_VERSION as SEMANTIC_SCHEMA_VERSION,
    ActivityObservation,
    AnalysisProvenance,
    CategoryObservation,
    PromotionDecision,
    PromotionRecommendation,
    SemanticCandidate,
    SemanticFingerprint,
    SemanticStatus,
    VisualEventCandidate,
    semantic_candidate_as_mapping,
)


SCHEMA_VERSION = 1
OPERATION = "vision.evidence.publish"
CAPABILITY = "vision.autonomous_promotion"
_TINY_JPEG = b"\xff\xd8\xff\xd9"


@dataclass
class _Queued:
    evidence_id: str
    candidate_id: str | None
    manifest: dict[str, Any]
    payload: bytes
    candidate: SemanticCandidate | None
    reason: str
    policy_version: str
    enqueued_at: datetime


class PromotionOutbox:
    """Bounded pending promotions. Clip files are not deleted from here."""

    def __init__(
        self,
        candidates: Any | None = None,
        *,
        max_pending: int = 8,
        max_age_s: float = 6 * 3600.0,
        clock: Callable[[], datetime] | None = None,
        client_runtime_version: str | None = None,
    ) -> None:
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError("promotion queue must keep at least one item")
        self.candidates = candidates
        self.max_pending = max_pending
        self.max_age_s = max_age_s
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.client_runtime_version = client_runtime_version
        self._items: dict[str, _Queued] = {}
        self._order: list[str] = []
        self._done: set[str] = set()
        self.released = 0
        self.rejected = 0
        self.transferred = 0

    def enqueue(
        self,
        decision: PolicyDecision,
        candidate: SemanticCandidate | None,
        manifest: Mapping[str, Any],
        payload: bytes,
    ) -> bool:
        """Queue one ``PROMOTE``. Any other decision is ignored."""
        if not isinstance(decision, PolicyDecision) or decision.decision != PromotionDecision.PROMOTE:
            return False
        evidence_id = str(manifest.get("evidence_id"))
        if evidence_id in self._done or evidence_id in self._items:
            return False
        if self.candidates is not None and decision.candidate_id:
            try:
                record = self.candidates.get(decision.candidate_id)
            except Exception:
                record = None
            if isinstance(record, dict) and record.get("upload_state") == "transferred":
                self._done.add(evidence_id)
                return False
            if isinstance(record, dict) and record.get("promotion_queue") in {"released", "rejected"}:
                return False
        self._make_room()
        self._items[evidence_id] = _Queued(
            evidence_id=evidence_id,
            candidate_id=decision.candidate_id,
            manifest=dict(manifest),
            payload=bytes(payload),
            candidate=candidate,
            reason=decision.reason,
            policy_version=decision.policy_version,
            enqueued_at=self.clock(),
        )
        self._order.append(evidence_id)
        self._mark(decision.candidate_id, "pending", hold=None)
        return True

    def pending_ids(self) -> list[str]:
        return list(self._order)

    def flush(self, publish: Callable[[dict[str, Any]], object]) -> int:
        """Send queued promotions. A transient failure stops this pass and keeps the item."""
        sent = 0
        for evidence_id in list(self._order):
            item = self._items.get(evidence_id)
            if item is None:
                continue
            captured_end = datetime.fromisoformat(str(item.manifest["captured_end"]))
            age = (self.clock() - captured_end).total_seconds()
            if age > self.max_age_s:
                self._drop(item, "released")
                continue
            envelope = _envelope_for(item, self.client_runtime_version)
            try:
                publish(envelope)
            except Exception as error:
                if getattr(error, "permanent", False):
                    self._drop(item, "rejected")
                    continue
                break
            self._ack(item)
            sent += 1
        return sent

    def recover(self, semantics: Any) -> int:
        """Queue stored ``PROMOTE`` decisions that have not left or been released."""
        if self.candidates is None:
            return 0
        added = 0
        for record in self.candidates.local_records():
            if record.get("upload_state") == "transferred":
                continue
            if record.get("promotion_queue") in {"released", "rejected"}:
                continue
            candidate_id = str(record.get("candidate_id"))
            try:
                path = semantics.directory(candidate_id) / "policy.json"
                if not path.is_file():
                    continue
                decision = parse_policy_decision(json.loads(path.read_text(encoding="utf-8")))
                if decision.decision != PromotionDecision.PROMOTE:
                    continue
                inspected = self.candidates.inspect(candidate_id)
                manifest = inspected.get("manifest")
                if not inspected.get("hash_ok") or not isinstance(manifest, Mapping):
                    continue
                candidate = semantics.load_candidate(candidate_id)
                payload = self.candidates.media(candidate_id)
            except Exception:
                continue
            if self.enqueue(decision, candidate, manifest, payload):
                added += 1
        return added

    def _make_room(self) -> None:
        while len(self._order) >= self.max_pending:
            evidence_id = self._order[0]
            item = self._items.get(evidence_id)
            if item is None:
                self._order.pop(0)
                continue
            self._drop(item, "released")

    def _ack(self, item: _Queued) -> None:
        self._done.add(item.evidence_id)
        self._remove(item.evidence_id)
        self.transferred += 1
        self._mark(item.candidate_id, "transferred", hold=None)

    def _drop(self, item: _Queued, hold: str) -> None:
        self._remove(item.evidence_id)
        if hold == "rejected":
            self.rejected += 1
        else:
            self.released += 1
        self._mark(item.candidate_id, "local_only", hold=hold)

    def _remove(self, evidence_id: str) -> None:
        self._items.pop(evidence_id, None)
        self._order = [item for item in self._order if item != evidence_id]

    def _mark(self, candidate_id: str | None, state: str, *, hold: str | None) -> None:
        if self.candidates is None or not candidate_id:
            return
        try:
            self.candidates.set_upload_state(candidate_id, state, now=self.clock(), hold=hold)
        except Exception:
            return


def build_promotion_envelope(
    *,
    embodiment_id: str,
    manifest: Mapping[str, Any],
    media: bytes,
    decision: str,
    reason: str,
    policy_version: str,
    semantic_candidate: Mapping[str, Any] | None = None,
    annotation: str = "omit",
    diagnostics: Mapping[str, str] | None = None,
    idempotency_key: str | None = None,
    schema_version: object = SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build one publish body. ``schema_version`` is an integer for a supported envelope."""
    body: dict[str, Any] = {
        "schema_version": schema_version,
        "embodiment_id": embodiment_id,
        "operation": OPERATION,
        "evidence": dict(manifest),
        "media_base64": base64.b64encode(media).decode("ascii"),
        "promotion": {
            "decision": decision,
            "reason": reason,
            "policy_version": policy_version,
        },
    }
    if annotation == "null":
        body["client_annotation"] = {"semantic_candidate": None}
    elif annotation == "value":
        body["client_annotation"] = {"semantic_candidate": None if semantic_candidate is None else dict(semantic_candidate)}
    if diagnostics:
        body["diagnostics"] = {key: value for key, value in diagnostics.items() if value}
    if idempotency_key is not None:
        body["idempotency_key"] = idempotency_key
    return body


def conformance_fixtures(
    embodiment_id: str,
    *,
    captured_at: datetime,
    camera_id: str = "camera:reference",
) -> dict[str, dict[str, Any]]:
    """Bodies a client can send. Invalid cases are present so a peer can reject them."""
    if captured_at.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    still_manifest, still_media = _still(embodiment_id, camera_id, captured_at, sequence=7)
    clip_manifest, clip_media = _clip(embodiment_id, camera_id, captured_at)
    still = build_promotion_envelope(
        embodiment_id=embodiment_id, manifest=still_manifest, media=still_media,
        decision="PROMOTE", reason="transition:empty->person_present", policy_version=POLICY_VERSION,
    )
    clip = build_promotion_envelope(
        embodiment_id=embodiment_id, manifest=clip_manifest, media=clip_media,
        decision="PROMOTE", reason="transition:empty->person_present", policy_version=POLICY_VERSION,
    )
    candidate = _annotation(clip_manifest, model_id="reference-perception")
    annotated = build_promotion_envelope(
        embodiment_id=embodiment_id, manifest=clip_manifest, media=clip_media,
        decision="PROMOTE", reason="transition:empty->person_present", policy_version=POLICY_VERSION,
        semantic_candidate=semantic_candidate_as_mapping(candidate), annotation="value",
        diagnostics={
            "candidate_id": candidate.candidate_id,
            "client_runtime_version": "reference-client",
            "perception_model_id": candidate.analysis.model_id if candidate.analysis else "",
            "perception_model_version": candidate.analysis.model_version if candidate.analysis else "",
        },
    )
    null_manifest, null_media = _still(embodiment_id, camera_id, captured_at, sequence=8)
    null_annotation = build_promotion_envelope(
        embodiment_id=embodiment_id, manifest=null_manifest, media=null_media,
        decision="PROMOTE", reason="transition:empty->person_present", policy_version=POLICY_VERSION,
        annotation="null",
    )
    invalid_hash = build_promotion_envelope(
        embodiment_id=embodiment_id, manifest=still_manifest, media=b"\xff\xd8\xff\xd0\xff\xd9",
        decision="PROMOTE", reason="transition:empty->person_present", policy_version=POLICY_VERSION,
    )
    unsupported = build_promotion_envelope(
        embodiment_id=embodiment_id, manifest=still_manifest, media=still_media,
        decision="PROMOTE", reason="transition:empty->person_present", policy_version=POLICY_VERSION,
        schema_version=2,
    )
    unsupported_text = build_promotion_envelope(
        embodiment_id=embodiment_id, manifest=still_manifest, media=still_media,
        decision="PROMOTE", reason="transition:empty->person_present", policy_version=POLICY_VERSION,
        schema_version="1",
    )
    retained = build_promotion_envelope(
        embodiment_id=embodiment_id, manifest=clip_manifest, media=clip_media,
        decision="RETAIN_LOCAL", reason="low_semantic_content", policy_version=POLICY_VERSION,
    )
    dropped = build_promotion_envelope(
        embodiment_id=embodiment_id, manifest=clip_manifest, media=clip_media,
        decision="DROP", reason="repetition:person_present", policy_version=POLICY_VERSION,
    )
    return {
        "still": still,
        "clip": clip,
        "annotated": annotated,
        "unannotated": clip,
        "null_annotation": null_annotation,
        "invalid_hash": invalid_hash,
        "unsupported_schema": unsupported,
        "unsupported_schema_text": unsupported_text,
        "retain_local": retained,
        "drop": dropped,
    }


def _envelope_for(item: _Queued, client_runtime_version: str | None) -> dict[str, Any]:
    diagnostics: dict[str, str] = {}
    if item.candidate_id:
        diagnostics["candidate_id"] = item.candidate_id
    if client_runtime_version:
        diagnostics["client_runtime_version"] = client_runtime_version
    candidate = item.candidate
    if candidate is not None and candidate.analysis is not None:
        diagnostics["perception_model_id"] = candidate.analysis.model_id
        diagnostics["perception_model_version"] = candidate.analysis.model_version
    mapping = None if candidate is None else semantic_candidate_as_mapping(candidate)
    return build_promotion_envelope(
        embodiment_id=str(item.manifest["embodiment_id"]),
        manifest=item.manifest,
        media=item.payload,
        decision="PROMOTE",
        reason=item.reason,
        policy_version=item.policy_version,
        semantic_candidate=mapping,
        annotation="omit" if mapping is None else "value",
        diagnostics=diagnostics,
    )


def _still(embodiment_id: str, camera_id: str, when: datetime, *, sequence: int) -> tuple[dict[str, Any], bytes]:
    stamp = when.isoformat(timespec="milliseconds")
    payload = _TINY_JPEG
    manifest = build_manifest(
        embodiment_id=embodiment_id, camera_id=camera_id, media_type="image",
        captured_start=stamp, captured_end=stamp, duration_ms=0,
        sequence_start=sequence, sequence_end=sequence, width=1, height=1,
        content_type=IMAGE_CONTENT_TYPE, sha256=_sha(payload), reason="visual_change",
    )
    return manifest, payload


def _clip(embodiment_id: str, camera_id: str, end: datetime) -> tuple[dict[str, Any], bytes]:
    start = end - timedelta(seconds=1)
    segments = (
        CapturedSegment(camera_id, 4, start, 1.0, 1, 1, "jpeg", IMAGE_CONTENT_TYPE, "local-4", _TINY_JPEG),
        CapturedSegment(camera_id, 5, end, 0.0, 1, 1, "jpeg", IMAGE_CONTENT_TYPE, "local-5", _TINY_JPEG),
    )
    payload = encode_clip(segments)
    manifest = build_manifest(
        embodiment_id=embodiment_id, camera_id=camera_id, media_type="video_clip",
        captured_start=start.isoformat(timespec="milliseconds"),
        captured_end=end.isoformat(timespec="milliseconds"),
        duration_ms=1000, sequence_start=4, sequence_end=5, width=1, height=1,
        content_type=CLIP_CONTENT_TYPE, sha256=_sha(payload), reason="visual_change",
    )
    return manifest, payload


def _annotation(manifest: Mapping[str, Any], *, model_id: str) -> SemanticCandidate:
    evidence_key = str(manifest["evidence_id"]).split(":", 1)[1]
    source = VisualEventCandidate(
        candidate_id="candidate:" + evidence_key,
        evidence_id=str(manifest["evidence_id"]),
        embodiment_id=str(manifest["embodiment_id"]),
        camera_id=str(manifest["camera_id"]),
        trigger_type="motion",
        trigger_started_at=str(manifest["captured_start"]),
        trigger_ended_at=str(manifest["captured_end"]),
        clip_start=str(manifest["captured_start"]),
        clip_end=str(manifest["captured_end"]),
        detector="frame_difference",
        detector_config={"sample_hz": 4},
        peak_score=0.2,
        average_score=0.1,
        sample_count=2,
        sequence_start=int(manifest["sequence_start"]),
        sequence_end=int(manifest["sequence_end"]),
        media_sha256=str(manifest["sha256"]),
    )
    observations = (CategoryObservation("person", 0.91),)
    activity = ActivityObservation("person_present", 0.91)
    analyzed = datetime.fromisoformat(str(manifest["captured_end"])) + timedelta(milliseconds=10)
    return SemanticCandidate(
        source=source,
        analyzed_at=analyzed.isoformat(timespec="milliseconds"),
        semantic_status=SemanticStatus.SUCCESS,
        analysis=AnalysisProvenance(model_id, "v1"),
        observations=observations,
        activity=activity,
        semantic_fingerprint=SemanticFingerprint(("person",), "person_present", "ab" * 32),
        promotion=PromotionRecommendation(PromotionDecision.PROMOTE, "transition:empty->person_present"),
    )


def _sha(payload: bytes) -> str:
    import hashlib
    return hashlib.sha256(payload).hexdigest()


# The semantic contract and this envelope share integer schema version 1.
assert SEMANTIC_SCHEMA_VERSION == SCHEMA_VERSION
