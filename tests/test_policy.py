"""Canonical promotion fixtures. No camera, model, or household graph."""
import socket
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from home_cortex_client.policy import (
    FORCE_PROMOTE,
    PolicyConfig,
    _failed,
    _success,
    canonical_capabilities,
    conformance_failures,
    decide,
    parse_policy_decision,
    policy_config,
    policy_decision_as_mapping,
    promotion_statistics,
    reference_cases,
    reference_suite,
    with_promotion,
)
from home_cortex_client.runtime import EdgeRuntime
from home_cortex_client.semantic import (
    PromotionDecision,
    SemanticContractError,
    SemanticFingerprint,
    SemanticStatus,
)
from home_cortex_client.sources import SyntheticCameraSource
from home_cortex_client.stream import StreamConfig


STAMP = "2026-10-04T18:00:00.000+00:00"
ROOT = Path(__file__).resolve().parents[1]


def _blocked(*_args, **_kwargs):
    raise AssertionError("promotion policy opened a network socket")


def test_reference_suite_is_hardware_independent(monkeypatch) -> None:
    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    claimed = canonical_capabilities(
        observe=True, observe_clip=True, semantic_filter=True, autonomous_promotion=True,
    )
    assert conformance_failures(decide, claimed) == []
    stats = promotion_statistics(reference_suite(decide))
    assert stats == {
        "count": 9,
        "decisions": {"PROMOTE": 4, "RETAIN_LOCAL": 5, "DROP": 0},
        "reasons": {
            "low_confidence": 1,
            "low_semantic_content": 1,
            "manual:force_promote": 1,
            "manual:force_retain": 1,
            "near_duplicate": 1,
            "new_category:package": 1,
            "semantic_unavailable": 1,
            "transition:empty->animal_present": 1,
            "transition:empty->person_present": 1,
        },
    }
    for _name, candidate, history, override, expect_decision, expect_reason in reference_cases():
        before = candidate.promotion
        decided = decide(candidate, history, decided_at=STAMP, override=override)
        assert candidate.promotion is before
        assert decided.decision.value == expect_decision
        assert decided.reason == expect_reason
        assert decided.candidate_id == candidate.candidate_id
        assert decided.evidence_id == candidate.evidence_id


def test_simple_camera_remains_valid_without_promotion() -> None:
    simple = canonical_capabilities(
        observe=True, observe_clip=False, semantic_filter=False, autonomous_promotion=False,
    )
    assert simple["vision.semantic_filter"] is False
    assert conformance_failures(decide, simple) == ["vision.autonomous_promotion is not claimed"]
    assert conformance_failures(decide, {"vision.observe": True}) == [
        "capabilities must declare every canonical name as a boolean",
    ]


def test_window_boundary_and_state_changes() -> None:
    person = _success("boundary-person", categories=(("person", 0.91),), activity=("person_present", 0.9))
    inside = _success(
        "boundary-inside", categories=(("person", 0.91),), activity=("person_present", 0.9), at=-8,
    )
    outside = _success(
        "boundary-outside", categories=(("person", 0.91),), activity=("person_present", 0.9), at=-8.1,
    )
    assert decide(person, (inside,), decided_at=STAMP).reason == "near_duplicate"
    assert decide(person, (outside,), decided_at=STAMP).reason == "transition:empty->person_present"

    entered = _success("entered", categories=(("person", 0.8),), activity=("person_entered", 0.8))
    repeated = decide(entered, (person,), decided_at=STAMP)
    assert repeated.decision == PromotionDecision.RETAIN_LOCAL
    assert repeated.reason == "repetition:person_present"

    left = _success("left", categories=(("person", 0.8),), activity=("person_left", 0.8))
    changed = decide(left, (person,), decided_at=STAMP)
    assert changed.decision == PromotionDecision.PROMOTE
    assert changed.reason == "significant_change"
    assert changed.coarse_state == "empty"


def test_opaque_hash_is_not_required_for_comparison() -> None:
    current = _success("hash-now", categories=(("person", 0.9),), activity=("person_present", 0.9))
    previous = _success("hash-then", categories=(("person", 0.9),), activity=("person_present", 0.9), at=-3)
    current = _with_hash(current, "a" * 64)
    previous = _with_hash(previous, "b" * 64)
    decided = decide(current, (previous,), decided_at=STAMP)
    assert decided.reason == "near_duplicate"
    assert decided.semantic_fingerprint is not None
    assert decided.semantic_fingerprint.opaque_hash == "a" * 64


def test_confidence_gate_is_uncalibrated_and_does_not_treat_zero_as_missing() -> None:
    exact = _success("exact", categories=(("person", 0.5),), activity=("person_present", 0.5))
    under = _success("under", categories=(("person", 0.5),), activity=("person_present", 0.49))
    zero = _success("zero", categories=(("person", 0.0),), activity=("person_present", 0.0))
    assert decide(exact, (), decided_at=STAMP).decision == PromotionDecision.PROMOTE
    assert decide(under, (), decided_at=STAMP).reason == "low_confidence"
    withheld = decide(zero, (), decided_at=STAMP)
    assert withheld.reason == "low_confidence"
    assert withheld.relevant_confidence == 0.0


def test_failures_history_and_manual_overrides_keep_evidence(tmp_path) -> None:
    sentinel = tmp_path / "clip.bin"
    sentinel.write_bytes(b"stage-2-evidence")
    person = _success("kept", categories=(("person", 0.91),), activity=("person_present", 0.9))
    earlier = _success(
        "kept-earlier", categories=(("person", 0.91),), activity=("person_present", 0.9), at=-2,
    )
    dropped = decide(
        person, (earlier,), PolicyConfig(repeat_decision=PromotionDecision.DROP), decided_at=STAMP,
    )
    assert dropped.decision == PromotionDecision.DROP
    assert dropped.reason == "near_duplicate"
    assert person.promotion is None
    assert sentinel.read_bytes() == b"stage-2-evidence"

    failed = _failed("broken", SemanticStatus.FAILED, "model_error")
    fallback = decide(failed, (), decided_at=STAMP)
    assert fallback.decision == PromotionDecision.RETAIN_LOCAL
    assert fallback.reason == "semantic_failed"
    assert fallback.semantic_fingerprint is None
    assert fallback.relevant_confidence is None
    forced = decide(failed, (), decided_at=STAMP, override=FORCE_PROMOTE)
    assert forced.decision == PromotionDecision.PROMOTE
    assert forced.reason == "manual:force_promote"
    assert forced.semantic_fingerprint is None
    assert with_promotion(failed, forced) is failed
    assert failed.observations == ()
    assert failed.promotion is None

    unavailable = _failed("quiet", SemanticStatus.UNAVAILABLE, "model_unavailable")
    visible = _success("after-failure", categories=(("animal", 0.8),))
    assert decide(visible, (unavailable, person), decided_at=STAMP).reason == "new_category:animal"
    promoted = decide(visible, (unavailable,), decided_at=STAMP)
    assert promoted.reason == "transition:empty->animal_present"
    assert decide(person, (person,), decided_at=STAMP).reason == "transition:empty->person_present"
    future = _success(
        "future", categories=(("person", 0.91),), activity=("person_present", 0.9), at=30,
    )
    assert decide(person, (future,), decided_at=STAMP).reason == "transition:empty->person_present"
    mystery = _success("mystery", categories=(("unknown_object", 0.95),))
    assert decide(mystery, (), decided_at=STAMP).reason == "low_semantic_content"
    assert decide(mystery, (), decided_at=STAMP).coarse_state == "unknown"


def test_policy_record_rejects_client_internals_and_clamps_time() -> None:
    person = _success("record", categories=(("person", 0.9),), activity=("person_present", 0.9))
    decided = decide(person, (), decided_at="2026-10-04T11:00:00+00:00")
    assert decided.decided_at == "2026-10-04T12:00:00.000+00:00"
    mapped = policy_decision_as_mapping(decided)
    mapped["camera_index"] = 0
    with pytest.raises(SemanticContractError):
        parse_policy_decision(mapped)
    del mapped["camera_index"]
    mapped["decision"] = 1
    with pytest.raises(SemanticContractError):
        parse_policy_decision(mapped)
    with pytest.raises(ValueError, match="force_promote or force_retain"):
        decide(person, (), decided_at=STAMP, override="force_drop")
    with pytest.raises(ValueError, match="retain or drop"):
        policy_config(repeat="delete")


def test_reference_module_has_no_hardware_vocabulary() -> None:
    text = (ROOT / "src" / "home_cortex_client" / "policy.py").read_text(encoding="utf-8").lower()
    for token in (
        "macbook", "avfoundation", "unitree", "optimus", "microduck", "yolo",
        "camera_index", "surreal", "opencv", "rk3566",
    ):
        assert token not in text
    assert "import perception" not in text
    assert "cv2" not in text


def test_local_capabilities_follow_runtime_facts(tmp_path) -> None:
    def build(name: str) -> EdgeRuntime:
        clock = lambda: datetime.now(timezone.utc)
        return EdgeRuntime(
            SyntheticCameraSource(fps=20, clock=clock),
            embodiment_id=name,
            config=StreamConfig(host="127.0.0.1", port=0),
            fps=20,
            evidence_dir=tmp_path / name / "evidence",
            candidate_dir=tmp_path / name / "candidates",
            clock=clock,
        )

    reference = build("embodiment:reference")
    other = build("embodiment:other-client")

    class _Analyzer:
        def stop(self) -> None:
            return None

    reference.attach_analyzer(_Analyzer())
    other.attach_analyzer(_Analyzer())
    assert reference.local_capabilities() == other.local_capabilities()
    assert reference.local_capabilities()["vision.autonomous_promotion"] is True
    assert reference.local_capabilities()["vision.observe"] is False
    assert reference.session_advertisement() == []
    assert other.session_advertisement() == []

    reference.start()
    try:
        deadline = time.time() + 2
        while not reference.camera_status()["available"] and time.time() < deadline:
            reference.wait(0.01)
        assert reference.camera_status()["available"] is True
        assert reference.session_advertisement() == ["vision.observe"]
        local = reference.local_capabilities()
        assert local["vision.observe"] is True
        assert local["vision.observe_clip"] is True
        assert local["vision.semantic_filter"] is True
        assert local["vision.autonomous_promotion"] is True
        assert "vision.autonomous_promotion" not in reference.session_advertisement()
        assert "vision.semantic_filter" not in reference.session_advertisement()
    finally:
        reference.stop()


def _with_hash(candidate, opaque: str):
    fingerprint = candidate.semantic_fingerprint
    return replace(candidate, semantic_fingerprint=SemanticFingerprint(
        fingerprint.categories, fingerprint.activity, opaque,
    ))
