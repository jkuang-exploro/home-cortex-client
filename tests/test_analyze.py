"""Local semantic filtering of Stage 2 clips. Models here are fakes."""
import hashlib
import json
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from home_cortex_client.analyze import (
    AnalysisDeferred,
    AnalysisFailure,
    FrameRead,
    SemanticWorker,
    analyze_candidate,
    map_labels,
    opaque_hash,
    peak_index_from_grids,
    sample_plan,
)
from home_cortex_client.buffer import RingBuffer
from home_cortex_client.config import ClientConfig
from home_cortex_client.debug import dispatch
from home_cortex_client.detect import MotionEvent
from home_cortex_client.candidates import CandidateEngine
from home_cortex_client.perception import VISION_FRAME_SWIFT, hog_confidence
from home_cortex_client.policy import PolicyConfig, parse_policy_decision, policy_decision_as_mapping
from home_cortex_client.runtime import EdgeRuntime
from home_cortex_client.semantic import (
    ActivityObservation,
    CategoryObservation,
    PromotionDecision,
    SemanticStatus,
    parse_semantic_candidate,
)
from home_cortex_client.sources import TINY_JPEG, SyntheticCameraSource
from home_cortex_client.stream import StreamConfig


START = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
CLOCK = lambda: datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)


class FakeModel:
    def __init__(self, reader=None, *, available=True, error=None):
        self.reader = reader
        self.model_id = "fake-filter"
        self.model_version = "test-1"
        self._available = available
        self.error = error
        self.calls = 0
        self.closed = False

    def available(self) -> bool:
        return self._available

    def close(self) -> None:
        self.closed = True

    def read_frame(self, jpeg: bytes) -> FrameRead:
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.reader is not None:
            return self.reader(self.calls, jpeg)
        return FrameRead(categories=())


def _candidate(tmp_path):
    buffer = RingBuffer(duration_s=30)
    for index in range(12):
        buffer.append_frame(
            camera_id="camera:built_in",
            captured_at=START + timedelta(seconds=index),
            width=1, height=1, payload=TINY_JPEG, duration_s=1,
        )
    engine = CandidateEngine(
        buffer,
        embodiment_id="embodiment:macbook-0",
        camera_id="camera:built_in",
        root=tmp_path,
    )
    engine.ingest(MotionEvent(
        START + timedelta(seconds=3), START + timedelta(seconds=5),
        peak_score=0.4, average_score=0.2, sample_count=9,
    ))
    engine.advance(START + timedelta(seconds=11))
    record = engine.store.local_records()[0]
    payload = engine.store.media(str(record["candidate_id"]))
    manifest = engine.store.inspect(str(record["candidate_id"]))["manifest"]
    return engine, record, payload, manifest


def _analyze(record, payload, manifest, model):
    return analyze_candidate(
        record, payload, manifest, model, clock=CLOCK, luma=lambda _jpeg: None,
    )


def test_sample_plan_is_deterministic_and_small() -> None:
    assert [index for index, _roles in sample_plan(10, 3)] == [0, 3, 5, 9]
    assert sample_plan(1, None) == ((0, ("start", "middle", "end")),)
    quiet = bytes(8)
    loud = bytes([255]) * 8
    assert peak_index_from_grids((quiet, quiet, loud)) == 2
    assert peak_index_from_grids((quiet, quiet, quiet)) == 1


def test_map_labels_keeps_coarse_categories_and_drops_identity() -> None:
    mapped = map_labels((
        ("Mr. Kuang", 0.99),
        ("Dylan", 0.99),
        ("Pu Ba", 0.99),
        ("outdoor", 0.9),
        ("night_sky", 0.5),
        ("dog", 0.8),
        ("Cat", 0.7),
        ("person", 0.4),
        ("person", 0.95),
        ("bottle", 0.6),
    ))
    assert mapped == (("animal", 0.8), ("food_or_drink", 0.6), ("person", 0.95))
    assert hog_confidence(0) == 0
    assert 0 < hog_confidence(1) < 1


def test_vision_compile_failure_stays_unavailable(tmp_path) -> None:
    from home_cortex_client.perception import VisionBackend

    backend = VisionBackend(cache_dir=tmp_path)
    backend._compile_failed = True
    assert backend.available() is False


def test_vision_helper_does_not_request_face_identity() -> None:
    assert "VNDetectHumanRectanglesRequest" in VISION_FRAME_SWIFT
    assert "VNRecognizeAnimalsRequest" in VISION_FRAME_SWIFT
    assert "VNClassifyImageRequest" in VISION_FRAME_SWIFT
    for banned in (
        "VNDetectFaceRectanglesRequest",
        "VNDetectFaceLandmarksRequest",
        "VNGenerateImageFeaturePrintRequest",
    ):
        assert banned not in VISION_FRAME_SWIFT


def test_success_empty_person_and_animal_keep_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network access"))
    engine, record, payload, manifest = _candidate(tmp_path)
    original = dict(record)

    empty, _trace = _analyze(record, payload, manifest, FakeModel())
    assert empty.semantic_status == SemanticStatus.SUCCESS
    assert empty.observations == ()
    assert empty.activity is None
    assert empty.semantic_fingerprint is not None
    assert empty.semantic_fingerprint.categories == ()
    assert empty.promotion is None
    assert empty.analysis is not None
    assert empty.analysis.model_id == "fake-filter"
    assert empty.analysis.model_version == "test-1"

    person, _trace = _analyze(
        record, payload, manifest,
        FakeModel(reader=lambda _call, _jpeg: FrameRead(
            categories=(("person", 0.91),), raw_labels=(("person", 0.91),),
        )),
    )
    assert [item.label for item in person.observations] == ["person"]
    assert person.activity is not None and person.activity.label == "person_present"
    assert person.semantic_fingerprint is not None
    assert person.semantic_fingerprint.categories == ("person",)
    assert person.semantic_fingerprint.activity == "person_present"
    assert person.semantic_fingerprint.opaque_hash == opaque_hash(
        "fake-filter", "test-1", person.observations, person.activity,
    )

    animal, trace = _analyze(
        record, payload, manifest,
        FakeModel(reader=lambda _call, _jpeg: FrameRead(
            categories=(("animal", 0.8), ("food_or_drink", 0.7)),
            raw_labels=(("dog", 0.8), ("bottle", 0.7)),
        )),
    )
    assert [item.label for item in animal.observations] == ["animal", "food_or_drink"]
    assert animal.activity is None
    assert animal.semantic_fingerprint is not None
    assert animal.semantic_fingerprint.categories == ("animal", "food_or_drink")
    assert trace["sampler"] == "start-peak-middle-end"
    assert engine.store.media(str(record["candidate_id"])) == payload
    assert engine.store.get(str(record["candidate_id"])) == original


def test_person_transition_and_low_information(tmp_path) -> None:
    _engine, record, payload, manifest = _candidate(tmp_path)

    def entered(call, _jpeg):
        if call == 1:
            return FrameRead(categories=())
        return FrameRead(categories=(("person", 0.88),))

    entered_result, _trace = _analyze(record, payload, manifest, FakeModel(reader=entered))
    assert entered_result.activity is not None
    assert entered_result.activity.label == "person_entered"

    def left(call, _jpeg):
        if call == 1:
            return FrameRead(categories=(("person", 0.7),))
        return FrameRead(categories=())

    left_result, _trace = _analyze(record, payload, manifest, FakeModel(reader=left))
    assert left_result.activity is not None
    assert left_result.activity.label == "person_left"

    weak, _trace = _analyze(
        record, payload, manifest,
        FakeModel(reader=lambda _call, _jpeg: FrameRead(categories=(("person", 0.1), ("bottle", 0.4)))),
    )
    assert weak.observations == ()
    assert weak.activity is None

    scene = dict(record)
    scene["scene_change_score"] = 0.5
    changed, _trace = _analyze(scene, payload, manifest, FakeModel())
    assert changed.activity is not None
    assert changed.activity.label == "large_scene_change"
    minor = dict(record)
    minor["scene_change_score"] = 0.1
    still, _trace = _analyze(minor, payload, manifest, FakeModel())
    assert still.activity is None


def test_fingerprint_repeats_for_the_same_model_output(tmp_path) -> None:
    _engine, record, payload, manifest = _candidate(tmp_path)
    model = FakeModel(reader=lambda _call, _jpeg: FrameRead(categories=(("person", 0.8), ("bottle", 0.6))))
    first, _trace = _analyze(record, payload, manifest, model)
    second, _trace = _analyze(record, payload, manifest, model)
    assert first.semantic_fingerprint == second.semantic_fingerprint
    assert first.semantic_fingerprint is not None
    assert first.semantic_fingerprint.opaque_hash != opaque_hash(
        "other-model", "test-1", first.observations, first.activity,
    )


def test_failures_keep_the_clip(tmp_path) -> None:
    engine, record, payload, manifest = _candidate(tmp_path)
    unavailable = FakeModel(available=False)
    with pytest.raises(AnalysisFailure) as missing:
        _analyze(record, payload, manifest, unavailable)
    assert missing.value.status == SemanticStatus.UNAVAILABLE
    assert missing.value.code == "model_unavailable"
    assert unavailable.calls == 0

    broken = FakeModel(error=AnalysisFailure(SemanticStatus.FAILED, "model_error"))
    with pytest.raises(AnalysisFailure) as failed:
        _analyze(record, payload, manifest, broken)
    assert failed.value.code == "model_error"

    with pytest.raises(AnalysisFailure) as mismatched:
        analyze_candidate(record, b"nope", None, FakeModel(), clock=CLOCK, luma=lambda _jpeg: None)
    assert mismatched.value.code == "hash_mismatch"

    garbage = b"not-a-clip"
    corrupt_record = dict(record)
    corrupt_record["media_sha256"] = hashlib.sha256(garbage).hexdigest()
    corrupt_manifest = dict(manifest)
    corrupt_manifest["sha256"] = corrupt_record["media_sha256"]
    with pytest.raises(AnalysisFailure) as corrupt:
        analyze_candidate(
            corrupt_record, garbage, corrupt_manifest, FakeModel(),
            clock=CLOCK, luma=lambda _jpeg: None,
        )
    assert corrupt.value.code == "corrupt_evidence"
    expired = dict(record)
    expired["status"] = "expired"
    with pytest.raises(AnalysisFailure) as gone:
        _analyze(expired, payload, manifest, FakeModel())
    assert gone.value.code == "evidence_unavailable"
    assert engine.store.media(str(record["candidate_id"])) == payload


def test_worker_stores_success_and_does_not_retry_it(tmp_path) -> None:
    engine, record, payload, _manifest = _candidate(tmp_path)
    model = FakeModel(reader=lambda _call, _jpeg: FrameRead(categories=(("person", 0.9),)))
    worker = SemanticWorker(
        engine.store, model, root=tmp_path / "semantics", hold=threading.Event(), clock=CLOCK,
    )
    assert worker.run_available() == 1
    candidate_id = str(record["candidate_id"])
    stored = worker.store.load_mapping(candidate_id)
    assert stored is not None
    parsed = parse_semantic_candidate(stored)
    assert parsed.semantic_status == SemanticStatus.SUCCESS
    assert parsed.promotion is not None
    assert parsed.promotion.decision == PromotionDecision.PROMOTE
    assert parsed.promotion.reason == "transition:empty->person_present"
    policy = parse_policy_decision(json.loads(
        (worker.store.directory(candidate_id) / "policy.json").read_text(encoding="utf-8")
    ))
    assert policy.decision == PromotionDecision.PROMOTE
    assert policy.reason == "transition:empty->person_present"
    assert policy.policy_version == "stage3-v1"
    assert "embodiment_id" not in policy_decision_as_mapping(policy)
    assert "camera_index" not in policy_decision_as_mapping(policy)
    assert parsed.analysis is not None and parsed.analysis.model_id == "fake-filter"
    assert parsed.embodiment_id == "embodiment:macbook-0"
    trace = json.loads((worker.store.directory(candidate_id) / "trace.json").read_text())
    assert trace["frames"]
    assert trace["accelerator"] == "not_recorded"
    assert worker.run_available() == 0
    assert engine.store.get(candidate_id)["upload_state"] == "local_only"
    assert engine.store.media(candidate_id) == payload


def test_worker_records_unavailable_error_and_hash_mismatch_without_deleting(tmp_path) -> None:
    engine, record, payload, _manifest = _candidate(tmp_path)
    candidate_id = str(record["candidate_id"])
    model = FakeModel(available=False)
    worker = SemanticWorker(
        engine.store, model, root=tmp_path / "semantics", hold=threading.Event(), clock=CLOCK,
    )
    assert worker.run_available() == 1
    unavailable = worker.store.load_mapping(candidate_id)
    assert unavailable is not None
    assert unavailable["semantic_status"] == "UNAVAILABLE"
    assert unavailable["failure_code"] == "model_unavailable"
    assert unavailable["observations"] == []
    assert unavailable["promotion"] is None
    assert unavailable["analysis"]["model_id"] == "fake-filter"
    unavailable_policy = _policy(worker, candidate_id)
    assert unavailable_policy.decision == PromotionDecision.RETAIN_LOCAL
    assert unavailable_policy.reason == "semantic_unavailable"
    assert unavailable_policy.semantic_fingerprint is None
    assert engine.store.media(candidate_id) == payload
    assert worker.run_available() == 0

    model._available = True
    model.reader = lambda _call, _jpeg: (_ for _ in ()).throw(RuntimeError("boom"))
    assert worker.run_available() == 1
    failed = worker.store.load_mapping(candidate_id)
    assert failed is not None
    assert failed["semantic_status"] == "FAILED"
    assert failed["failure_code"] == "model_error"
    assert failed["promotion"] is None
    failed_policy = _policy(worker, candidate_id)
    assert failed_policy.reason == "semantic_failed"
    assert failed_policy.semantic_fingerprint is None
    assert engine.store.media(candidate_id) == payload

    path = Path(str(engine.store.inspect(candidate_id)["provenance"]["local_clip_path"]))
    path.write_bytes(b"tampered")
    # Settled failures are not retried; remove the result to recheck the bytes.
    for name in ("semantic.json", "trace.json"):
        (worker.store.directory(candidate_id) / name).unlink()
    assert worker.run_available() == 1
    mismatched = worker.store.load_mapping(candidate_id)
    assert mismatched is not None
    assert mismatched["failure_code"] == "hash_mismatch"
    assert mismatched["promotion"] is None
    assert _policy(worker, candidate_id).reason == "semantic_failed"
    assert path.read_bytes() == b"tampered"
    assert engine.store.get(candidate_id)["status"] == "local"


def test_worker_waits_for_interactive_observation(tmp_path) -> None:
    engine, _record, _payload, _manifest = _candidate(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def reader(_call, _jpeg):
        started.set()
        assert release.wait(2)
        return FrameRead(categories=())

    hold = threading.Event()
    hold.set()
    worker = SemanticWorker(
        engine.store, FakeModel(reader=reader), root=tmp_path / "semantics", hold=hold, clock=CLOCK,
    )
    thread = threading.Thread(target=worker.run_available)
    thread.start()
    began = time.perf_counter()
    time.sleep(0.05)
    waited = time.perf_counter() - began
    assert waited < 0.3
    assert not started.is_set()
    hold.clear()
    assert started.wait(1)
    release.set()
    thread.join(2)
    assert not thread.is_alive()


def test_sampling_pauses_between_frames_while_observation_holds(tmp_path) -> None:
    _engine, record, payload, manifest = _candidate(tmp_path)
    hold = threading.Event()
    calls = []

    def reader(_call, _jpeg):
        calls.append(_call)
        if len(calls) == 1:
            hold.set()
        return FrameRead(categories=())

    box: list[object] = []

    def analyze() -> None:
        box.append(analyze_candidate(
            record, payload, manifest, FakeModel(reader=reader),
            clock=CLOCK, luma=lambda _jpeg: None, hold=hold,
        ))

    thread = threading.Thread(target=analyze)
    thread.start()
    deadline = time.time() + 1
    while len(calls) < 1 and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)
    assert calls == [1]
    hold.clear()
    thread.join(2)
    assert not thread.is_alive()
    assert len(calls) >= 2
    assert box


def test_stop_during_hold_leaves_the_candidate_queued(tmp_path) -> None:
    engine, record, _payload, _manifest = _candidate(tmp_path)
    hold = threading.Event()
    hold.set()
    worker = SemanticWorker(
        engine.store, FakeModel(), root=tmp_path / "semantics", hold=hold, clock=CLOCK,
    )
    worker._stop.set()
    assert worker.run_available() == 0
    assert worker.store.load_mapping(str(record["candidate_id"])) is None


def test_runtime_does_not_start_an_analyzer_by_itself(tmp_path) -> None:
    runtime = EdgeRuntime(
        SyntheticCameraSource(fps=20),
        config=StreamConfig(host="127.0.0.1", port=0),
        fps=20,
        evidence_dir=tmp_path / "evidence",
        candidate_dir=tmp_path / "candidates",
    )
    assert runtime.semantic_index() == {"analyzer": "off", "results": []}
    runtime.start()
    try:
        assert not any(thread.name == "semantic-analyzer" for thread in threading.enumerate())
    finally:
        runtime.stop()


def test_debug_semantics_route_reads_local_results(tmp_path) -> None:
    runtime = EdgeRuntime(
        SyntheticCameraSource(),
        evidence_dir=tmp_path / "evidence",
        candidate_dir=tmp_path / "candidates",
    )
    buffer = runtime.buffer
    for index in range(12):
        buffer.append_frame(
            camera_id="camera:built_in",
            captured_at=START + timedelta(seconds=index),
            width=1, height=1, payload=TINY_JPEG, duration_s=1,
        )
    runtime.candidates.ingest(MotionEvent(
        START + timedelta(seconds=3), START + timedelta(seconds=5),
        peak_score=0.4, average_score=0.2, sample_count=9,
    ))
    runtime.candidates.advance(START + timedelta(seconds=11))
    model = FakeModel(reader=lambda _call, _jpeg: FrameRead(categories=(("animal", 0.66),)))
    worker = SemanticWorker(
        runtime.candidates.store, model,
        root=tmp_path / "candidates-semantics", hold=threading.Event(), clock=CLOCK,
    )
    runtime.attach_analyzer(worker)
    assert worker.run_available() == 1
    status, _kind, _headers, body = dispatch(runtime, "GET", "/debug/semantics", {})
    listed = json.loads(body)
    assert status == 200
    assert listed["analyzer"] == "fake-filter"
    assert listed["results"][0]["categories"] == ["animal"]
    assert listed["results"][0]["promotion_decision"] == "PROMOTE"
    assert listed["results"][0]["promotion_reason"] == "transition:empty->animal_present"
    candidate_id = listed["results"][0]["candidate_id"]
    status, _kind, _headers, body = dispatch(runtime, "GET", "/debug/semantics/" + candidate_id, {})
    assert status == 200
    shown = json.loads(body)
    assert shown["semantic_status"] == "SUCCESS"
    assert shown["promotion"]["decision"] == "PROMOTE"
    capabilities = dispatch(runtime, "GET", "/debug/capabilities", {})
    declared = json.loads(capabilities[3])
    assert declared["session_advertisement"] == []
    assert declared["local_capabilities"]["vision.autonomous_promotion"] is True
    assert declared["local_capabilities"]["vision.semantic_filter"] is True
    assert "vision.autonomous_promotion" not in declared["session_advertisement"]
    missing = dispatch(runtime, "GET", "/debug/semantics/candidate:missing", {})
    assert missing[0] == 404


def test_analyzer_configuration_defaults_off(monkeypatch) -> None:
    assert ClientConfig().analyzer == "off"
    assert ClientConfig().promotion_window_s == 8.0
    assert ClientConfig().promotion_min_confidence == 0.5
    assert ClientConfig().promotion_repeat == "retain"
    assert ClientConfig().promotion_override is None
    for name in (
        "HOME_CORTEX_CLIENT_ANALYZER",
        "HOME_CORTEX_CLIENT_PROMOTION_WINDOW_SECONDS",
        "HOME_CORTEX_CLIENT_PROMOTION_MIN_CONFIDENCE",
        "HOME_CORTEX_CLIENT_PROMOTION_REPEAT",
        "HOME_CORTEX_CLIENT_PROMOTION_OVERRIDE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME_CORTEX_CLIENT_ANALYZER", "HOG")
    assert ClientConfig.from_env().analyzer == "hog"
    monkeypatch.setenv("HOME_CORTEX_CLIENT_PROMOTION_REPEAT", "drop")
    monkeypatch.setenv("HOME_CORTEX_CLIENT_PROMOTION_OVERRIDE", "force_retain")
    selected = ClientConfig.from_env()
    assert selected.promotion_repeat == "drop"
    assert selected.promotion_override == "force_retain"
    monkeypatch.setenv("HOME_CORTEX_CLIENT_ANALYZER", "sometimes")
    with pytest.raises(ValueError, match="ANALYZER"):
        ClientConfig.from_env()
    monkeypatch.setenv("HOME_CORTEX_CLIENT_ANALYZER", "off")
    monkeypatch.setenv("HOME_CORTEX_CLIENT_PROMOTION_REPEAT", "maybe")
    with pytest.raises(ValueError, match="PROMOTION_REPEAT"):
        ClientConfig.from_env()


def test_repeat_drop_keeps_both_clips(tmp_path) -> None:
    buffer = RingBuffer(duration_s=40)
    for index in range(26):
        buffer.append_frame(
            camera_id="camera:built_in",
            captured_at=START + timedelta(seconds=index),
            width=1, height=1, payload=TINY_JPEG, duration_s=1,
        )
    engine = CandidateEngine(
        buffer,
        embodiment_id="embodiment:macbook-0",
        camera_id="camera:built_in",
        root=tmp_path,
    )
    engine.ingest(MotionEvent(
        START + timedelta(seconds=3), START + timedelta(seconds=5),
        peak_score=0.4, average_score=0.2, sample_count=9,
    ))
    engine.ingest(MotionEvent(
        START + timedelta(seconds=15), START + timedelta(seconds=17),
        peak_score=0.4, average_score=0.2, sample_count=9,
    ))
    engine.advance(START + timedelta(seconds=25))
    records = engine.store.local_records()
    assert len(records) == 2
    payloads = {
        str(record["candidate_id"]): engine.store.media(str(record["candidate_id"]))
        for record in records
    }
    worker = SemanticWorker(
        engine.store,
        FakeModel(reader=lambda _call, _jpeg: FrameRead(categories=(("person", 0.9),))),
        root=tmp_path / "semantics",
        hold=threading.Event(),
        clock=CLOCK,
        policy=PolicyConfig(repeat_decision=PromotionDecision.DROP),
    )
    assert worker.run_available() == 2
    reasons = []
    for record in records:
        candidate_id = str(record["candidate_id"])
        parsed = parse_semantic_candidate(worker.store.load_mapping(candidate_id))
        policy = _policy(worker, candidate_id)
        assert parsed.promotion is not None
        assert parsed.promotion.decision == policy.decision
        reasons.append(policy.reason)
        assert engine.store.media(candidate_id) == payloads[candidate_id]
        assert engine.store.get(candidate_id)["upload_state"] == "local_only"
    assert reasons == ["transition:empty->person_present", "near_duplicate"]
    assert _policy(worker, str(records[1]["candidate_id"])).decision == PromotionDecision.DROP


def test_worker_override_retains_without_deleting(tmp_path) -> None:
    engine, record, payload, _manifest = _candidate(tmp_path)
    candidate_id = str(record["candidate_id"])
    worker = SemanticWorker(
        engine.store,
        FakeModel(reader=lambda _call, _jpeg: FrameRead(categories=(("person", 0.9),))),
        root=tmp_path / "semantics",
        hold=threading.Event(),
        clock=CLOCK,
        override="force_retain",
    )
    assert worker.run_available() == 1
    parsed = parse_semantic_candidate(worker.store.load_mapping(candidate_id))
    assert parsed.promotion is not None
    assert parsed.promotion.reason == "manual:force_retain"
    assert parsed.observations
    assert engine.store.media(candidate_id) == payload
    assert engine.store.get(candidate_id)["upload_state"] == "local_only"


def _policy(worker: SemanticWorker, candidate_id: str):
    return parse_policy_decision(json.loads(
        (worker.store.directory(candidate_id) / "policy.json").read_text(encoding="utf-8")
    ))


def test_import_analyze_does_not_load_opencv() -> None:
    script = (
        "import sys; "
        "import home_cortex_client.analyze, home_cortex_client.perception; "
        "assert 'cv2' not in sys.modules"
    )
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-c", script], cwd=root / "src", check=True)


def test_deferred_analysis_is_distinct_from_failure() -> None:
    assert AnalysisDeferred.__name__ == "AnalysisDeferred"
    observations = (CategoryObservation("person", 0.5),)
    activity = ActivityObservation("person_present", 0.5)
    assert opaque_hash("m", "1", observations, activity) == opaque_hash("m", "1", observations, activity)
