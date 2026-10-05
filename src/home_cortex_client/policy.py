"""Client-agnostic promotion policy for canonical Stage 2/3 records.

The policy sees ``SemanticCandidate`` values, recent history, and configuration.
It does not see a camera, an accelerator, or a vendor perception object.
``PROMOTE`` means the clip deserves later Home Cortex processing. It does not
establish a household fact, send a notification, or upload bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Callable, Mapping, Sequence

from .semantic import (
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
)


POLICY_VERSION = "stage3-v1"
COARSE_STATES = frozenset({
    "empty", "person_present", "animal_present", "mixed_activity", "unknown",
})
CAPABILITIES = (
    "vision.observe",
    "vision.observe_clip",
    "vision.semantic_filter",
    "vision.autonomous_promotion",
)
FORCE_PROMOTE = "force_promote"
FORCE_RETAIN = "force_retain"
_TRANSITIONS = (
    ("empty", "person_present"),
    ("empty", "animal_present"),
    ("empty", "mixed_activity"),
)
_KNOWN = frozenset({"empty", "person_present", "animal_present", "mixed_activity"})
_PERSON_ACTIVITIES = frozenset({"person_present", "person_entered", "person_left"})
_FIELDS = frozenset({
    "policy_version", "decision", "reason", "decided_at", "candidate_id",
    "evidence_id", "semantic_fingerprint", "relevant_confidence", "coarse_state",
})
_BASE = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class PolicyConfig:
    """Reference thresholds. Confidence is an uncalibrated gate, not a shared probability."""

    policy_version: str = POLICY_VERSION
    repeat_window_s: float = 8.0
    minimum_promote_confidence: float = 0.5
    repeat_decision: PromotionDecision = PromotionDecision.RETAIN_LOCAL
    transitions: tuple[tuple[str, str], ...] = _TRANSITIONS
    promote_new_category: bool = True
    promote_state_change: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise ValueError("policy_version must be nonempty")
        if isinstance(self.repeat_window_s, bool) or not isinstance(self.repeat_window_s, (int, float)):
            raise ValueError("repeat_window_s must be a finite duration")
        if not math.isfinite(float(self.repeat_window_s)) or float(self.repeat_window_s) < 0:
            raise ValueError("repeat_window_s must be a finite duration")
        if isinstance(self.minimum_promote_confidence, bool) or not isinstance(
            self.minimum_promote_confidence, (int, float)
        ):
            raise ValueError("minimum_promote_confidence must be within [0, 1]")
        if not 0 <= float(self.minimum_promote_confidence) <= 1:
            raise ValueError("minimum_promote_confidence must be within [0, 1]")
        if self.repeat_decision not in {PromotionDecision.RETAIN_LOCAL, PromotionDecision.DROP}:
            raise ValueError("repetition must retain or drop")
        for previous, current in self.transitions:
            if previous not in COARSE_STATES or current not in COARSE_STATES or previous == current:
                raise ValueError("transitions must name two different coarse states")
        if not isinstance(self.promote_new_category, bool) or not isinstance(self.promote_state_change, bool):
            raise ValueError("promotion switches must be true or false")


@dataclass(frozen=True)
class PolicyDecision:
    """Canonical promotion metadata. Constructing it does not upload or delete evidence."""

    decision: PromotionDecision
    reason: str
    policy_version: str
    decided_at: str
    candidate_id: str
    evidence_id: str
    coarse_state: str
    semantic_fingerprint: SemanticFingerprint | None = None
    relevant_confidence: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, PromotionDecision):
            raise SemanticContractError("unknown promotion decision")
        PromotionRecommendation(self.decision, self.reason)
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise SemanticContractError("policy_version must be nonempty")
        _time(self.decided_at, "decided_at")
        _identifier(self.candidate_id, "candidate")
        _identifier(self.evidence_id, "evidence")
        if self.coarse_state not in COARSE_STATES:
            raise SemanticContractError("unknown coarse state")
        if self.semantic_fingerprint is not None and not isinstance(
            self.semantic_fingerprint, SemanticFingerprint
        ):
            raise SemanticContractError("semantic_fingerprint must be canonical")
        if self.relevant_confidence is not None:
            _unit(self.relevant_confidence)

    @property
    def recommendation(self) -> PromotionRecommendation:
        return PromotionRecommendation(self.decision, self.reason)


def policy_config(
    *,
    repeat_window_s: float = 8.0,
    minimum_promote_confidence: float = 0.5,
    repeat: str = "retain",
) -> PolicyConfig:
    """Build the reference configuration from client-local settings."""
    if repeat == "retain":
        decision = PromotionDecision.RETAIN_LOCAL
    elif repeat == "drop":
        decision = PromotionDecision.DROP
    else:
        raise ValueError("repetition must retain or drop")
    return PolicyConfig(
        repeat_window_s=repeat_window_s,
        minimum_promote_confidence=minimum_promote_confidence,
        repeat_decision=decision,
    )


def canonical_capabilities(
    *,
    observe: bool,
    observe_clip: bool,
    semantic_filter: bool,
    autonomous_promotion: bool,
) -> dict[str, bool]:
    """Declare support explicitly. Embodiment hardware does not imply a capability."""
    flags = {
        "vision.observe": observe,
        "vision.observe_clip": observe_clip,
        "vision.semantic_filter": semantic_filter,
        "vision.autonomous_promotion": autonomous_promotion,
    }
    if set(flags) != set(CAPABILITIES) or any(not isinstance(value, bool) for value in flags.values()):
        raise ValueError("capabilities must be the canonical booleans")
    return flags


def coarse_state(candidate: SemanticCandidate) -> str:
    """End-of-clip state. Failure and unclassified content stay ``unknown``."""
    if not isinstance(candidate, SemanticCandidate):
        raise SemanticContractError("coarse state requires a semantic candidate")
    if candidate.semantic_status != SemanticStatus.SUCCESS:
        return "unknown"
    labels = {item.label for item in candidate.observations}
    activity = None if candidate.activity is None else candidate.activity.label
    person = "person" in labels or activity in {"person_present", "person_entered"}
    if activity == "person_left":
        person = False
    animal = "animal" in labels
    other = labels - {"person", "animal", "unknown_object"}
    if person and (animal or other):
        return "mixed_activity"
    if animal and other:
        return "mixed_activity"
    if person:
        return "person_present"
    if animal:
        return "animal_present"
    if other:
        return "mixed_activity"
    if "unknown_object" in labels or activity in {"unknown_activity", "large_scene_change", "object_activity"}:
        return "unknown"
    return "empty"


def decide(
    candidate: SemanticCandidate,
    history: Sequence[SemanticCandidate] = (),
    config: PolicyConfig | None = None,
    *,
    decided_at: str,
    override: str | None = None,
) -> PolicyDecision:
    """Choose one promotion outcome. The candidate record is not modified."""
    if not isinstance(candidate, SemanticCandidate):
        raise SemanticContractError("policy input must be a semantic candidate")
    settings = config or PolicyConfig()
    if not isinstance(settings, PolicyConfig):
        raise SemanticContractError("policy configuration must be canonical")
    for item in history:
        if not isinstance(item, SemanticCandidate):
            raise SemanticContractError("history must contain semantic candidates")
    state = coarse_state(candidate)
    stamp = _decision_time(decided_at, candidate.analyzed_at)
    if override == FORCE_PROMOTE:
        return _decision(candidate, settings, PromotionDecision.PROMOTE, "manual:force_promote", stamp, state, _min_confidence(candidate))
    if override == FORCE_RETAIN:
        return _decision(candidate, settings, PromotionDecision.RETAIN_LOCAL, "manual:force_retain", stamp, state, _min_confidence(candidate))
    if override is not None:
        raise ValueError("manual override must be force_promote or force_retain")
    if candidate.semantic_status == SemanticStatus.UNAVAILABLE:
        return _decision(candidate, settings, PromotionDecision.RETAIN_LOCAL, "semantic_unavailable", stamp, state, None)
    if candidate.semantic_status != SemanticStatus.SUCCESS:
        return _decision(candidate, settings, PromotionDecision.RETAIN_LOCAL, "semantic_failed", stamp, state, None)
    recent = _recent(history, candidate, settings.repeat_window_s)
    previous = recent[-1] if recent else None
    previous_state = "empty" if previous is None else coarse_state(previous)
    chosen, reason, confidence = _select(candidate, previous, previous_state, recent, state, settings)
    if chosen == PromotionDecision.PROMOTE and not _confident(confidence, settings):
        chosen = PromotionDecision.RETAIN_LOCAL
        reason = "low_confidence"
    return _decision(candidate, settings, chosen, reason, stamp, state, confidence)


def with_promotion(candidate: SemanticCandidate, decision: PolicyDecision) -> SemanticCandidate:
    """Attach the advisory pair only to a successful analysis."""
    if candidate.semantic_status != SemanticStatus.SUCCESS:
        return candidate
    if candidate.candidate_id != decision.candidate_id or candidate.evidence_id != decision.evidence_id:
        raise SemanticContractError("promotion metadata must name the same evidence")
    return replace(candidate, promotion=decision.recommendation)


def promotion_statistics(decisions: Sequence[PolicyDecision]) -> dict[str, object]:
    counts = {item.value: 0 for item in PromotionDecision}
    reasons: dict[str, int] = {}
    for item in decisions:
        counts[item.decision.value] += 1
        reasons[item.reason] = reasons.get(item.reason, 0) + 1
    return {"count": len(decisions), "decisions": counts, "reasons": dict(sorted(reasons.items()))}


def policy_decision_as_mapping(value: PolicyDecision) -> dict[str, object]:
    if not isinstance(value, PolicyDecision):
        raise SemanticContractError("expected PolicyDecision")
    fingerprint = value.semantic_fingerprint
    return {
        "policy_version": value.policy_version,
        "decision": value.decision.value,
        "reason": value.reason,
        "decided_at": value.decided_at,
        "candidate_id": value.candidate_id,
        "evidence_id": value.evidence_id,
        "coarse_state": value.coarse_state,
        "semantic_fingerprint": None if fingerprint is None else {
            "categories": list(fingerprint.categories),
            "activity": fingerprint.activity,
            "opaque_hash": fingerprint.opaque_hash,
        },
        "relevant_confidence": value.relevant_confidence,
    }


def parse_policy_decision(value: Mapping[str, object]) -> PolicyDecision:
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise SemanticContractError("policy decision fields do not match the canonical result")
    if any(type(value[name]) is not str for name in (
        "decision", "reason", "policy_version", "decided_at",
        "candidate_id", "evidence_id", "coarse_state",
    )):
        raise SemanticContractError("policy decision text fields must be strings")
    fingerprint = value["semantic_fingerprint"]
    parsed: SemanticFingerprint | None
    if fingerprint is None:
        parsed = None
    elif isinstance(fingerprint, Mapping) and set(fingerprint) == {"categories", "activity", "opaque_hash"}:
        categories = fingerprint["categories"]
        activity = fingerprint["activity"]
        opaque = fingerprint["opaque_hash"]
        if not isinstance(categories, list) or any(type(label) is not str for label in categories):
            raise SemanticContractError("semantic_fingerprint categories must be a list of labels")
        if activity is not None and type(activity) is not str:
            raise SemanticContractError("semantic_fingerprint activity must be text or null")
        if opaque is not None and type(opaque) is not str:
            raise SemanticContractError("semantic_fingerprint opaque_hash must be text or null")
        parsed = SemanticFingerprint(tuple(categories), activity, opaque)
    else:
        raise SemanticContractError("semantic_fingerprint must be canonical")
    try:
        return PolicyDecision(
            decision=PromotionDecision(value["decision"]),
            reason=value["reason"],
            policy_version=value["policy_version"],
            decided_at=value["decided_at"],
            candidate_id=value["candidate_id"],
            evidence_id=value["evidence_id"],
            coarse_state=value["coarse_state"],
            semantic_fingerprint=parsed,
            relevant_confidence=value["relevant_confidence"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, SemanticContractError):
            raise
        raise SemanticContractError(f"invalid policy decision: {error}") from error


def conformance_failures(
    policy: Callable[..., PolicyDecision],
    capabilities: Mapping[str, bool],
) -> list[str]:
    """Canonical checks for a client that claims ``vision.autonomous_promotion``."""
    failures: list[str] = []
    if set(capabilities) != set(CAPABILITIES) or any(type(value) is not bool for value in capabilities.values()):
        failures.append("capabilities must declare every canonical name as a boolean")
        return failures
    if capabilities["vision.autonomous_promotion"] is not True:
        failures.append("vision.autonomous_promotion is not claimed")
        return failures
    settings = PolicyConfig()
    for name, candidate, history, override, expect_decision, expect_reason in reference_cases():
        try:
            decided = policy(
                candidate, history, settings,
                decided_at="2026-10-04T18:00:00.000+00:00", override=override,
            )
        except Exception as error:
            failures.append(f"{name}: {type(error).__name__}: {error}")
            continue
        if decided.decision.value != expect_decision or decided.reason != expect_reason:
            failures.append(
                f"{name}: expected {expect_decision} {expect_reason}, "
                f"got {decided.decision.value} {decided.reason}"
            )
        if decided.policy_version != POLICY_VERSION:
            failures.append(f"{name}: policy_version")
        if decided.candidate_id != candidate.candidate_id or decided.evidence_id != candidate.evidence_id:
            failures.append(f"{name}: evidence provenance changed")
        if candidate.semantic_status != SemanticStatus.SUCCESS and decided.semantic_fingerprint is not None:
            failures.append(f"{name}: failed analysis claimed a fingerprint")
        if candidate.semantic_status == SemanticStatus.SUCCESS and decided.decision == PromotionDecision.PROMOTE:
            if decided.semantic_fingerprint is None:
                failures.append(f"{name}: promotion omitted the fingerprint")
        try:
            mapped = policy_decision_as_mapping(decided)
            if parse_policy_decision(json.loads(json.dumps(mapped))) != decided:
                failures.append(f"{name}: policy json did not round-trip")
            if set(mapped) - _FIELDS:
                failures.append(f"{name}: non-canonical policy fields")
        except SemanticContractError as error:
            failures.append(f"{name}: {error}")
    other = _success(
        "same-scene",
        categories=(("person", 0.91),),
        activity=("person_present", 0.9),
        embodiment_id="embodiment:other-client",
        camera_id="camera:other",
        detector="other-detector",
        model_id="other-perception",
    )
    reference = _success("same-scene", categories=(("person", 0.91),), activity=("person_present", 0.9))
    try:
        left = policy(reference, (), settings, decided_at="2026-10-04T18:00:00.000+00:00", override=None)
        right = policy(other, (), settings, decided_at="2026-10-04T18:00:00.000+00:00", override=None)
        if (left.decision, left.reason) != (right.decision, right.reason):
            failures.append("client identity changed the promotion result")
    except Exception as error:
        failures.append(f"cross-client fixture: {error}")
    return failures


def reference_suite(
    policy: Callable[..., PolicyDecision],
    config: PolicyConfig | None = None,
    *,
    decided_at: str = "2026-10-04T18:00:00.000+00:00",
) -> list[PolicyDecision]:
    """Run the nine canonical fixtures. The callable sees only contract objects."""
    settings = config or PolicyConfig()
    return [
        policy(candidate, history, settings, decided_at=decided_at, override=override)
        for _name, candidate, history, override, _decision, _reason in reference_cases()
    ]


def reference_cases():
    empty = _success("empty-now", categories=(), at=0)
    earlier_empty = _success("empty-earlier", categories=(), at=-2)
    person = _success("person-now", categories=(("person", 0.91),), activity=("person_present", 0.9), at=0)
    person_earlier = _success("person-earlier", categories=(("person", 0.91),), activity=("person_present", 0.9), at=-8)
    added = _success(
        "person-package",
        categories=(("person", 0.91), ("package", 0.8)),
        activity=("person_present", 0.9),
        at=0,
    )
    animal = _success("animal-now", categories=(("animal", 0.88),), at=0)
    weak = _success("weak-person", categories=(("person", 0.2),), activity=("person_present", 0.2), at=0)
    unavailable = _failed("unavailable", SemanticStatus.UNAVAILABLE, "model_unavailable")
    return (
        ("empty → empty", empty, (earlier_empty,), None, "RETAIN_LOCAL", "low_semantic_content"),
        ("empty → person", person, (earlier_empty,), None, "PROMOTE", "transition:empty->person_present"),
        ("person → same state", person, (person_earlier,), None, "RETAIN_LOCAL", "near_duplicate"),
        ("person → new category", added, (person_earlier,), None, "PROMOTE", "new_category:package"),
        ("new animal", animal, (earlier_empty,), None, "PROMOTE", "transition:empty->animal_present"),
        ("low confidence", weak, (), None, "RETAIN_LOCAL", "low_confidence"),
        ("semantic unavailable", unavailable, (), None, "RETAIN_LOCAL", "semantic_unavailable"),
        ("manual promotion", empty, (), FORCE_PROMOTE, "PROMOTE", "manual:force_promote"),
        ("manual retention", person, (earlier_empty,), FORCE_RETAIN, "RETAIN_LOCAL", "manual:force_retain"),
    )


def _select(candidate, previous, previous_state, recent, state, settings: PolicyConfig):
    for origin, destination in settings.transitions:
        if previous_state == origin and state == destination:
            return PromotionDecision.PROMOTE, f"transition:{origin}->{destination}", _support(candidate, None)
    if settings.promote_new_category:
        seen = set().union(*({item.label for item in prior.observations} for prior in recent)) if recent else set()
        novel = sorted(label for label in {item.label for item in candidate.observations}
                       if label not in seen and label != "unknown_object")
        if novel:
            return PromotionDecision.PROMOTE, f"new_category:{novel[0]}", _support(candidate, {novel[0]})
    if (settings.promote_state_change and previous is not None
            and previous_state in _KNOWN and state in _KNOWN and previous_state != state):
        return PromotionDecision.PROMOTE, "significant_change", _support(candidate, None)
    # Empty and unknown are not repetitions of a promoted state.
    if previous is not None and previous_state == state and state in _KNOWN and state != "empty":
        reason = "near_duplicate" if _same_symbols(previous, candidate) else f"repetition:{state}"
        return settings.repeat_decision, reason, _min_confidence(candidate)
    return PromotionDecision.RETAIN_LOCAL, "low_semantic_content", _min_confidence(candidate)


def _support(candidate: SemanticCandidate, labels: set[str] | None) -> float | None:
    scores: list[float] = []
    for item in candidate.observations:
        if labels is None or item.label in labels:
            scores.append(item.confidence)
    if labels is None and candidate.activity is not None:
        scores.append(candidate.activity.confidence)
    if not scores:
        return None
    return min(scores)


def _min_confidence(candidate: SemanticCandidate) -> float | None:
    if candidate.semantic_status != SemanticStatus.SUCCESS:
        return None
    return _support(candidate, None)


def _confident(confidence: float | None, settings: PolicyConfig) -> bool:
    return confidence is not None and confidence >= settings.minimum_promote_confidence


def _same_symbols(previous: SemanticCandidate, current: SemanticCandidate) -> bool:
    left = previous.semantic_fingerprint
    right = current.semantic_fingerprint
    return (
        left is not None and right is not None
        and left.categories == right.categories
        and left.activity == right.activity
    )


def _recent(history: Sequence[SemanticCandidate], current: SemanticCandidate, window_s: float):
    current_at = datetime.fromisoformat(current.analyzed_at)
    chosen = []
    for item in history:
        if item.candidate_id == current.candidate_id or item.semantic_status != SemanticStatus.SUCCESS:
            continue
        then = datetime.fromisoformat(item.analyzed_at)
        if then <= current_at and (current_at - then).total_seconds() <= window_s:
            chosen.append(item)
    chosen.sort(key=lambda item: item.analyzed_at)
    return chosen


def _decision(candidate, settings, decision, reason, decided_at, state, confidence) -> PolicyDecision:
    fingerprint = candidate.semantic_fingerprint if candidate.semantic_status == SemanticStatus.SUCCESS else None
    if decision != PromotionDecision.PROMOTE:
        confidence = confidence if candidate.semantic_status == SemanticStatus.SUCCESS else None
    return PolicyDecision(
        decision=decision,
        reason=reason,
        policy_version=settings.policy_version,
        decided_at=decided_at,
        candidate_id=candidate.candidate_id,
        evidence_id=candidate.evidence_id,
        coarse_state=state,
        semantic_fingerprint=fingerprint,
        relevant_confidence=confidence,
    )


def _decision_time(decided_at: str, analyzed_at: str) -> str:
    _time(decided_at, "decided_at")
    chosen = datetime.fromisoformat(decided_at)
    analyzed = datetime.fromisoformat(analyzed_at)
    if chosen < analyzed:
        chosen = analyzed
    return chosen.isoformat(timespec="milliseconds")


def _success(
    name: str,
    *,
    categories: tuple[tuple[str, float], ...],
    activity: tuple[str, float] | None = None,
    at: float = 0,
    embodiment_id: str = "embodiment:reference",
    camera_id: str = "camera:reference",
    detector: str = "reference-detector",
    model_id: str = "reference-perception",
) -> SemanticCandidate:
    observations = tuple(CategoryObservation(label, score) for label, score in categories)
    activity_value = None if activity is None else ActivityObservation(activity[0], activity[1])
    return _candidate(
        name,
        status=SemanticStatus.SUCCESS,
        observations=observations,
        activity=activity_value,
        fingerprint=fingerprint_for(observations, activity_value),
        at=at,
        embodiment_id=embodiment_id,
        camera_id=camera_id,
        detector=detector,
        model_id=model_id,
        failure_code=None,
    )


def _failed(name: str, status: SemanticStatus, code: str) -> SemanticCandidate:
    return _candidate(
        name, status=status, observations=(), activity=None, fingerprint=None,
        at=0, embodiment_id="embodiment:reference", camera_id="camera:reference",
        detector="reference-detector", model_id="reference-perception", failure_code=code,
    )


def _candidate(
    name: str,
    *,
    status: SemanticStatus,
    observations: tuple[CategoryObservation, ...],
    activity: ActivityObservation | None,
    fingerprint: SemanticFingerprint | None,
    at: float,
    embodiment_id: str,
    camera_id: str,
    detector: str,
    model_id: str,
    failure_code: str | None,
) -> SemanticCandidate:
    end = _BASE + timedelta(seconds=at)
    start = end - timedelta(seconds=2)
    trigger = end - timedelta(seconds=1)
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    source = VisualEventCandidate(
        candidate_id="candidate:" + digest[:32],
        evidence_id="evidence:" + digest[32:],
        embodiment_id=embodiment_id,
        camera_id=camera_id,
        trigger_type="motion",
        trigger_started_at=trigger.isoformat(),
        trigger_ended_at=trigger.isoformat(),
        clip_start=start.isoformat(),
        clip_end=end.isoformat(),
        detector=detector,
        detector_config={"sample_hz": 1},
        peak_score=0.2,
        average_score=0.1,
        sample_count=2,
        sequence_start=1,
        sequence_end=2,
        media_sha256=digest,
    )
    return SemanticCandidate(
        source=source,
        analyzed_at=end.isoformat(),
        semantic_status=status,
        analysis=AnalysisProvenance(model_id, "canonical-1"),
        observations=observations,
        activity=activity,
        semantic_fingerprint=fingerprint,
        failure_code=failure_code,
    )


def _time(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise SemanticContractError(f"{name} must be a timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise SemanticContractError(f"{name} must be a timezone-aware timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SemanticContractError(f"{name} must be a timezone-aware timestamp")


def _identifier(value: str, prefix: str) -> None:
    token = value.split(":", 1)[1] if value.startswith(prefix + ":") else ""
    if not token or any(character not in "0123456789abcdefABCDEF" for character in token):
        # Stage 2 ids are hex, but the contract also allows other id bodies.
        body = value[len(prefix) + 1:] if value.startswith(prefix + ":") else ""
        if not body or any(not (character.isalnum() or character in "_-") for character in body):
            raise SemanticContractError(f"{prefix} id is invalid")


def _unit(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SemanticContractError("relevant_confidence must be within [0, 1]")
    if not 0 <= float(value) <= 1:
        raise SemanticContractError("relevant_confidence must be within [0, 1]")
