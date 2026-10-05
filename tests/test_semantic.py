"""Stage 3 annotation contracts preserve Stage 2 evidence without side effects."""
import json
import math
import socket
from datetime import datetime, timedelta, timezone

import pytest

from home_cortex_client.buffer import RingBuffer
from home_cortex_client.candidates import CandidateEngine
from home_cortex_client.detect import MotionEvent
from home_cortex_client.semantic import (
    ActivityObservation,
    AnalysisProvenance,
    CategoryObservation,
    PromotionDecision,
    PromotionRecommendation,
    SemanticCandidate,
    SemanticContractError,
    SemanticFingerprint,
    SemanticStatus,
    VisualEventCandidate,
    fingerprint_for,
    parse_semantic_candidate,
    semantic_candidate_as_mapping,
)
from home_cortex_client.sources import TINY_JPEG


START = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
ANALYZED_AT = (START + timedelta(seconds=12)).isoformat()


def _stage2_record(tmp_path):
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
    return engine, engine.store.local_records()[0]


def _success(source):
    observations = (
        CategoryObservation("person", 0.94),
        CategoryObservation("package", 0.76),
    )
    activity = ActivityObservation("person_present", 0.9)
    return SemanticCandidate(
        source=source,
        analyzed_at="2026-10-03T12:00:12+00:00",
        semantic_status=SemanticStatus.SUCCESS,
        analysis=AnalysisProvenance("local-vision", "1.2"),
        observations=observations,
        activity=activity,
        semantic_fingerprint=fingerprint_for(observations, activity),
        promotion=PromotionRecommendation(
            PromotionDecision.RETAIN_LOCAL, "near_duplicate",
        ),
    )


def test_stage2_candidate_to_semantic_candidate_preserves_evidence(tmp_path, monkeypatch) -> None:
    engine, record = _stage2_record(tmp_path)
    original = dict(record)
    media = engine.store.media(record["candidate_id"])
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("database/network access"))

    source = VisualEventCandidate.from_stage2(record)
    semantic = _success(source)
    serialized = json.loads(json.dumps(semantic_candidate_as_mapping(semantic)))

    assert semantic.candidate_id == record["candidate_id"]
    assert semantic.evidence_id == record["evidence_id"]
    assert semantic.embodiment_id == record["embodiment_id"]
    assert source.media_sha256 == record["media_sha256"]
    assert serialized["source"]["clip_start"] == record["clip_start"]
    assert serialized["source"]["peak_score"] == record["peak_score"]
    assert serialized["source"]["detector_config"] == record["detector_config"]
    assert serialized["source"]["sequence_start"] == record["sequence_start"]
    assert serialized["semantic_fingerprint"] == {
        "categories": ["package", "person"], "activity": "person_present", "opaque_hash": None,
    }
    assert serialized["promotion"] == {
        "decision": "RETAIN_LOCAL", "reason": "near_duplicate",
    }
    assert parse_semantic_candidate(serialized) == semantic
    assert record == original
    assert engine.store.media(record["candidate_id"]) == media


@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf, True, "0.9"])
def test_confidence_rejects_invalid_values(value) -> None:
    with pytest.raises(SemanticContractError, match="confidence"):
        CategoryObservation("person", value)
    with pytest.raises(SemanticContractError, match="confidence"):
        ActivityObservation("person_present", value)


def test_confidence_accepts_bounds_and_is_distinct_from_physical_p95() -> None:
    assert CategoryObservation("person", 0).confidence == 0
    assert ActivityObservation("person_present", 1).confidence == 1
    with pytest.raises(TypeError):
        CategoryObservation("person", 0.8, p95=0.1)


def test_unknown_and_named_identity_labels_are_rejected() -> None:
    for label in ("unknown_label", "Mr. Kuang", "Dylan"):
        with pytest.raises(SemanticContractError, match="category"):
            CategoryObservation(label, 0.9)
    with pytest.raises(SemanticContractError, match="activity"):
        ActivityObservation("security_issue", 0.7)


def test_missing_analysis_and_analyzer_failure_are_explicit(tmp_path) -> None:
    _engine, record = _stage2_record(tmp_path)
    source = VisualEventCandidate.from_stage2(record)
    with pytest.raises(SemanticContractError, match="requires analysis"):
        SemanticCandidate(source, ANALYZED_AT, SemanticStatus.SUCCESS)

    for status, code in ((SemanticStatus.UNAVAILABLE, "model_unavailable"),
                         (SemanticStatus.FAILED, "inference_failed")):
        annotation = SemanticCandidate(
            source, ANALYZED_AT, status, failure_code=code,
        )
        assert parse_semantic_candidate(
            json.loads(json.dumps(semantic_candidate_as_mapping(annotation)))
        ) == annotation
        assert annotation.evidence_id == record["evidence_id"]
        assert annotation.promotion is None
    with pytest.raises(SemanticContractError, match="cannot claim"):
        SemanticCandidate(
            source, ANALYZED_AT, SemanticStatus.FAILED,
            observations=(CategoryObservation("person", 0.9),),
            failure_code="inference_failed",
        )


def test_fingerprint_validation_and_recommendation_are_non_authoritative(tmp_path) -> None:
    engine, record = _stage2_record(tmp_path)
    source = VisualEventCandidate.from_stage2(record)
    with pytest.raises(SemanticContractError, match="sorted and unique"):
        SemanticFingerprint(("person", "animal"), None)
    with pytest.raises(SemanticContractError, match="fingerprint must match"):
        SemanticCandidate(
            source, ANALYZED_AT, SemanticStatus.SUCCESS,
            analysis=AnalysisProvenance("model", "1"),
            observations=(CategoryObservation("person", 0.8),),
            semantic_fingerprint=SemanticFingerprint(("animal",), None),
        )
    for decision in PromotionDecision:
        candidate = SemanticCandidate(
            source, ANALYZED_AT, SemanticStatus.SUCCESS,
            analysis=AnalysisProvenance("model", "1"),
            semantic_fingerprint=SemanticFingerprint((), None),
            promotion=PromotionRecommendation(decision, "low_semantic_content"),
        )
        assert parse_semantic_candidate(semantic_candidate_as_mapping(candidate)) == candidate
    assert engine.store.media(record["candidate_id"])
    fingerprint = SemanticFingerprint(("person",), "person_present", "a" * 64)
    observations = (CategoryObservation("person", 0.8),)
    candidate = SemanticCandidate(
        source, ANALYZED_AT, SemanticStatus.SUCCESS,
        analysis=AnalysisProvenance("model", "1"),
        observations=observations,
        activity=ActivityObservation("person_present", 0.7),
        semantic_fingerprint=fingerprint,
    )
    assert parse_semantic_candidate(semantic_candidate_as_mapping(candidate)) == candidate
    with pytest.raises(SemanticContractError, match="machine-readable"):
        PromotionRecommendation(PromotionDecision.PROMOTE, "interesting!")
    with pytest.raises(SemanticContractError, match="machine-readable"):
        PromotionRecommendation(PromotionDecision.PROMOTE, "important")


def test_serialization_rejects_unknown_authority_fields_and_bad_provenance(tmp_path) -> None:
    _engine, record = _stage2_record(tmp_path)
    mapping = semantic_candidate_as_mapping(_success(VisualEventCandidate.from_stage2(record)))
    mapping["person_present"] = True
    with pytest.raises(SemanticContractError, match="unknown semantic candidate fields"):
        parse_semantic_candidate(mapping)
    del mapping["person_present"]
    mapping["source"]["evidence_id"] = "visual_candidate:wrong-kind"
    with pytest.raises(SemanticContractError, match="evidence_id"):
        parse_semantic_candidate(mapping)
    mapping["source"]["evidence_id"] = record["evidence_id"]
    mapping["schema_version"] = 2
    with pytest.raises(SemanticContractError, match="schema version"):
        parse_semantic_candidate(mapping)
