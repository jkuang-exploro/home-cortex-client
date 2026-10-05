"""Canonical promotion transport. Device identity is an id, not a code branch."""
from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import pytest

from home_cortex_client.analyze import FrameRead, SemanticWorker
from home_cortex_client.backend import BackendSession, PromotionTransportError
from home_cortex_client.buffer import RingBuffer
from home_cortex_client.candidates import CandidateEngine, CandidateStore, candidate_id_for
from home_cortex_client.detect import MotionEvent
from home_cortex_client.evidence import build_manifest
from home_cortex_client.policy import POLICY_VERSION, PolicyDecision
from home_cortex_client.publish import (
    OPERATION,
    SCHEMA_VERSION,
    PromotionOutbox,
    conformance_fixtures,
)
from home_cortex_client.runtime import EdgeRuntime
from home_cortex_client.semantic import PromotionDecision
from home_cortex_client.sources import TINY_JPEG, SyntheticCameraSource
from home_cortex_client.stream import StreamConfig


NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
REFERENCE = "embodiment:reference"
MACBOOK = "embodiment:macbook-0"


class _Down(Exception):
    permanent = False
    code = "transport_failure"


class _Rejected(Exception):
    def __init__(self) -> None:
        self.permanent = True
        self.code = "hash_mismatch"
        super().__init__("hash_mismatch")


class _Model:
    model_id = "reference-perception"
    model_version = "v1"

    def available(self) -> bool:
        return True

    def close(self) -> None:
        return None

    def read_frame(self, _jpeg: bytes) -> FrameRead:
        return FrameRead(categories=(("person", 0.91),), raw_labels=(("person", 0.91),))


def test_conformance_shape_is_the_same_for_two_clients(monkeypatch) -> None:
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network access"))
    left = conformance_fixtures(REFERENCE, captured_at=NOW)
    right = conformance_fixtures("embodiment:other-client", captured_at=NOW)
    assert set(left) == {
        "still", "clip", "annotated", "unannotated", "null_annotation",
        "invalid_hash", "unsupported_schema", "unsupported_schema_text",
        "retain_local", "drop",
    }
    assert set(left) == set(right)
    for name in left:
        assert set(left[name]) == set(right[name])
        assert left[name]["operation"] == right[name]["operation"] == OPERATION
        assert left[name]["schema_version"] == right[name]["schema_version"]
    assert left["still"]["schema_version"] == SCHEMA_VERSION
    assert isinstance(left["still"]["schema_version"], int)
    assert left["unsupported_schema"]["schema_version"] == 2
    assert left["unsupported_schema_text"]["schema_version"] == "1"
    assert left["annotated"]["client_annotation"]["semantic_candidate"]["semantic_status"] == "SUCCESS"
    assert "client_annotation" not in left["unannotated"]
    assert left["null_annotation"]["client_annotation"] == {"semantic_candidate": None}
    assert left["retain_local"]["promotion"]["decision"] == "RETAIN_LOCAL"
    assert left["drop"]["promotion"]["decision"] == "DROP"
    assert left["still"]["evidence"]["reason"] == "visual_change"
    assert left["clip"]["evidence"]["media_type"] == "video_clip"


def test_macbook_and_reference_clients_post_the_same_envelope(monkeypatch) -> None:
    posted = []

    def fake_urlopen(request, timeout):
        assert timeout == 60
        posted.append((request.full_url, json.loads(request.data)))
        return BytesIO(json.dumps({"verified": True}).encode())

    monkeypatch.setattr("home_cortex_client.backend.urlopen", fake_urlopen)
    for embodiment_id in (MACBOOK, REFERENCE):
        session = BackendSession("http://cortex.local", "test-key", embodiment_id)
        session.session_id = "runtime-session:1"
        envelope = conformance_fixtures(embodiment_id, captured_at=NOW)["annotated"]
        session.publish_evidence(envelope)
    assert posted[0][0].endswith("/v1/embodiments/embodiment:macbook-0/session/evidence")
    assert posted[1][0].endswith("/v1/embodiments/embodiment:reference/session/evidence")
    assert set(posted[0][1]) == set(posted[1][1])
    assert posted[0][1]["operation"] == posted[1][1]["operation"] == OPERATION
    assert posted[0][1]["promotion"] == posted[1][1]["promotion"]
    assert posted[0][1]["schema_version"] == 1
    mac_text = json.dumps(posted[0][1])
    for token in ("camera_index", "avfoundation", "unitree", "optimus", "yolo"):
        assert token not in mac_text


def test_outbox_sends_only_promote_and_retries_without_duplicates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network access"))
    store = CandidateStore(tmp_path, max_items=8, max_age_s=3600, max_bytes=8_000_000)
    outbox = PromotionOutbox(store, max_pending=2, clock=lambda: NOW)
    queued = [_save(store, sequence) for sequence in (1, 2, 3)]
    retain = _decision(queued[0][0], queued[0][1], PromotionDecision.RETAIN_LOCAL, "low_semantic_content", "empty")
    assert outbox.enqueue(retain, None, queued[0][0], TINY_JPEG) is False
    dropped = _decision(queued[0][0], queued[0][1], PromotionDecision.DROP, "repetition:person_present", "person_present")
    assert outbox.enqueue(dropped, None, queued[0][0], TINY_JPEG) is False
    assert outbox.pending_ids() == []

    assert outbox.enqueue(_promote(queued[0]), None, queued[0][0], TINY_JPEG) is True
    assert outbox.enqueue(_promote(queued[0]), None, queued[0][0], TINY_JPEG) is False
    assert outbox.enqueue(_promote(queued[1]), None, queued[1][0], TINY_JPEG) is True
    assert outbox.enqueue(_promote(queued[2]), None, queued[2][0], TINY_JPEG) is True
    assert outbox.released == 1
    assert outbox.pending_ids() == [queued[1][0]["evidence_id"], queued[2][0]["evidence_id"]]
    for _manifest, candidate_id in queued:
        assert (store._directory(candidate_id) / "payload").is_file()

    calls = {"n": 0}

    def flaky(envelope):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _Down("offline")
        return {"verified": True, "evidence": envelope["evidence"]}

    assert outbox.flush(flaky) == 0
    assert len(outbox.pending_ids()) == 2
    assert outbox.flush(flaky) == 2
    assert calls["n"] == 3
    assert outbox.flush(lambda _body: pytest.fail("duplicate upload")) == 0
    assert outbox.pending_ids() == []
    assert outbox.transferred == 2
    assert store.get(queued[1][1])["upload_state"] == "transferred"
    assert store.get(queued[0][1])["promotion_queue"] == "released"
    assert (store._directory(queued[0][1]) / "payload").is_file()


def test_permanent_rejection_is_not_retried(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network access"))
    store = CandidateStore(tmp_path, max_items=4, max_age_s=3600, max_bytes=8_000_000)
    outbox = PromotionOutbox(store, clock=lambda: NOW)
    manifest, candidate_id = _save(store, 4)
    assert outbox.enqueue(_promote((manifest, candidate_id)), None, manifest, TINY_JPEG) is True
    calls = {"n": 0}

    def reject(_envelope):
        calls["n"] += 1
        raise _Rejected()

    assert outbox.flush(reject) == 0
    assert outbox.flush(reject) == 0
    assert calls["n"] == 1
    assert outbox.rejected == 1
    assert outbox.pending_ids() == []
    assert (store._directory(candidate_id) / "payload").is_file()
    assert store.get(candidate_id)["upload_state"] == "local_only"


def test_publish_transport_classifies_disconnect_and_hash_mismatch(monkeypatch) -> None:
    def raise_offline(_request, timeout):
        from urllib.error import URLError
        raise URLError("down")

    monkeypatch.setattr("home_cortex_client.backend.urlopen", raise_offline)
    session = BackendSession("http://cortex.local", "test-key", REFERENCE)
    session.session_id = "runtime-session:1"
    envelope = conformance_fixtures(REFERENCE, captured_at=NOW)["still"]
    with pytest.raises(PromotionTransportError) as offline:
        session.publish_evidence(envelope)
    assert offline.value.permanent is False

    def raise_hash(_request, timeout):
        from urllib.error import HTTPError
        body = json.dumps({"error": {"code": "hash_mismatch", "message": "media bytes do not match"}}).encode()
        raise HTTPError("http://cortex.local", 422, "hash_mismatch", {}, BytesIO(body))

    monkeypatch.setattr("home_cortex_client.backend.urlopen", raise_hash)
    with pytest.raises(PromotionTransportError) as rejected:
        session.publish_evidence(envelope)
    assert rejected.value.permanent is True
    assert rejected.value.code == "hash_mismatch"


def test_worker_queues_a_promoted_clip_and_leaves_bytes_in_place(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network access"))
    engine = _engine(tmp_path, REFERENCE)
    outbox = PromotionOutbox(engine.store, clock=lambda: NOW)
    worker = SemanticWorker(
        engine.store, _Model(), root=tmp_path / "semantics", hold=__import__("threading").Event(),
        clock=lambda: NOW, on_promoted=outbox.enqueue,
    )
    assert worker.run_available() == 1
    assert len(outbox.pending_ids()) == 1
    candidate_id = outbox.pending_ids() and engine.store.local_records()[0]["candidate_id"]
    assert engine.store.get(str(candidate_id))["upload_state"] == "pending"
    sent = []
    assert outbox.flush(sent.append) == 1
    assert sent[0]["operation"] == OPERATION
    assert sent[0]["promotion"]["decision"] == "PROMOTE"
    assert sent[0]["promotion"]["policy_version"] == POLICY_VERSION
    assert sent[0]["schema_version"] == 1
    assert sent[0]["diagnostics"]["perception_model_id"] == "reference-perception"
    assert sent[0]["client_annotation"]["semantic_candidate"]["source"]["embodiment_id"] == REFERENCE
    record = engine.store.get(str(candidate_id))
    assert record["upload_state"] == "transferred"
    assert engine.store.media(str(candidate_id))
    assert outbox.flush(sent.append) == 0
    assert len(sent) == 1


def test_session_advertisement_follows_configured_names(tmp_path) -> None:
    def build(name: str) -> EdgeRuntime:
        clock = lambda: datetime.now(timezone.utc)
        runtime = EdgeRuntime(
            SyntheticCameraSource(fps=20, clock=clock),
            embodiment_id=name,
            config=StreamConfig(host="127.0.0.1", port=0),
            fps=20,
            evidence_dir=tmp_path / name / "evidence",
            candidate_dir=tmp_path / name / "candidates",
            clock=clock,
        )
        runtime.set_session_capabilities((
            "vision.observe", "vision.observe_clip", "vision.autonomous_promotion",
        ))
        return runtime

    reference = build(REFERENCE)
    other = build("embodiment:other-client")

    class _Analyzer:
        def stop(self) -> None:
            return None

    reference.attach_analyzer(_Analyzer())
    other.attach_analyzer(_Analyzer())
    assert reference.session_advertisement() == ["vision.autonomous_promotion"]
    assert other.session_advertisement() == reference.session_advertisement()
    reference.start()
    try:
        deadline = datetime.now(timezone.utc).timestamp() + 2
        while not reference.camera_status()["available"] and datetime.now(timezone.utc).timestamp() < deadline:
            reference.wait(0.01)
        assert reference.session_advertisement() == [
            "vision.observe", "vision.observe_clip", "vision.autonomous_promotion",
        ]
    finally:
        reference.stop()
    with pytest.raises(ValueError):
        reference.set_session_capabilities(("vision.observe", "audio.speak"))


def test_publisher_source_has_no_client_branch() -> None:
    text = (Path(__file__).resolve().parents[1] / "src" / "home_cortex_client" / "publish.py").read_text(encoding="utf-8").lower()
    for token in (
        "macbook", "avfoundation", "unitree", "optimus", "microduck", "yolo",
        "camera_index", "surreal", "opencv", "rk3566", "cv2",
    ):
        assert token not in text
    assert "home_cortex." not in text
    assert "import home_cortex" not in text


def _engine(tmp_path: Path, embodiment_id: str) -> CandidateEngine:
    buffer = RingBuffer(duration_s=30)
    start = NOW - timedelta(seconds=20)
    for index in range(12):
        buffer.append_frame(
            camera_id="camera:reference",
            captured_at=start + timedelta(seconds=index),
            width=1, height=1, payload=TINY_JPEG, duration_s=1,
        )
    engine = CandidateEngine(
        buffer, embodiment_id=embodiment_id, camera_id="camera:reference", root=tmp_path / "candidates",
    )
    engine.ingest(MotionEvent(
        start + timedelta(seconds=3), start + timedelta(seconds=5),
        peak_score=0.4, average_score=0.2, sample_count=9,
    ))
    engine.advance(start + timedelta(seconds=11))
    return engine


def _save(store: CandidateStore, sequence: int) -> tuple[dict, str]:
    stamp = NOW.isoformat(timespec="milliseconds")
    manifest = build_manifest(
        embodiment_id=REFERENCE, camera_id="camera:reference", media_type="image",
        captured_start=stamp, captured_end=stamp, duration_ms=0,
        sequence_start=sequence, sequence_end=sequence, width=1, height=1,
        content_type="image/jpeg", sha256="a" * 64, reason="visual_change",
    )
    fields = {
        "embodiment_id": REFERENCE,
        "camera_id": "camera:reference",
        "trigger_type": "motion",
        "trigger_started_at": stamp,
        "trigger_ended_at": stamp,
        "clip_start": stamp,
        "clip_end": stamp,
        "peak_score": 0.2,
        "evidence_id": manifest["evidence_id"],
    }
    candidate_id = candidate_id_for(fields)
    record = {
        **fields,
        "candidate_id": candidate_id,
        "upload_state": "local_only",
        "status": "local",
        "retained_at": stamp,
    }
    store.save(record, manifest, TINY_JPEG, now=NOW)
    return manifest, candidate_id


def _promote(saved: tuple[dict, str]) -> PolicyDecision:
    return _decision(saved[0], saved[1], PromotionDecision.PROMOTE, "transition:empty->person_present", "person_present")


def _decision(manifest, candidate_id, decision, reason, state) -> PolicyDecision:
    return PolicyDecision(
        decision=decision,
        reason=reason,
        policy_version=POLICY_VERSION,
        decided_at="2026-10-04T12:00:00.010+00:00",
        candidate_id=candidate_id,
        evidence_id=str(manifest["evidence_id"]),
        coarse_state=state,
        relevant_confidence=0.91,
    )
