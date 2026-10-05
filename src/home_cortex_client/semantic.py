"""Provisional, device-local annotations of Stage 2 visual-change candidates.

This module has no model, transport, or database dependency. In particular,
``VisualEventCandidate`` is a local changed-scene clip, not Home Cortex's
persistent ``VisualCandidate`` identity anchor.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


SCHEMA_VERSION = 1
CATEGORY_LABELS = frozenset({
    "person", "animal", "vehicle", "package", "door", "furniture",
    "screen", "food_or_drink", "unknown_object",
})
ACTIVITY_LABELS = frozenset({
    "person_present", "person_entered", "person_left", "object_activity",
    "large_scene_change", "unknown_activity",
})
_CODE = re.compile(r"[a-z][a-z0-9_]*(?::[a-z0-9_]+(?:->[a-z0-9_]+)?)?\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_VAGUE_REASONS = frozenset({"important", "interesting"})


class SemanticContractError(ValueError):
    """The annotation is malformed or claims more authority than this contract allows."""


class SemanticStatus(StrEnum):
    SUCCESS = "SUCCESS"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"


class PromotionDecision(StrEnum):
    PROMOTE = "PROMOTE"
    RETAIN_LOCAL = "RETAIN_LOCAL"
    DROP = "DROP"


def _id(value: object, prefix: str, name: str) -> str:
    if (not isinstance(value, str) or not value.startswith(prefix + ":")
            or not re.fullmatch(r"[A-Za-z0-9_-]+", value[len(prefix) + 1:])):
        raise SemanticContractError(f"{name} must be a {prefix}: ID")
    return value


def _time(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise SemanticContractError(f"{name} must be a timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise SemanticContractError(f"{name} must be a timezone-aware timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SemanticContractError(f"{name} must be a timezone-aware timestamp")
    return value


def _confidence(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise SemanticContractError(f"{name} must be finite and within [0, 1]")
    return float(value)


def _positive_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticContractError(f"{name} must be nonempty text")
    return value


def _code(value: object, name: str) -> str:
    if not isinstance(value, str) or not _CODE.fullmatch(value) or value in _VAGUE_REASONS:
        raise SemanticContractError(f"{name} must be a machine-readable code")
    return value


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticContractError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class VisualEventCandidate:
    """Immutable provenance copied from one Stage 2 ``candidate:`` record."""

    candidate_id: str
    evidence_id: str
    embodiment_id: str
    camera_id: str
    trigger_type: str
    trigger_started_at: str
    trigger_ended_at: str
    clip_start: str
    clip_end: str
    detector: str
    detector_config: Mapping[str, bool | int | float]
    peak_score: float
    average_score: float
    sample_count: int
    sequence_start: int
    sequence_end: int
    media_sha256: str

    def __post_init__(self) -> None:
        _id(self.candidate_id, "candidate", "candidate_id")
        _id(self.evidence_id, "evidence", "evidence_id")
        _id(self.embodiment_id, "embodiment", "embodiment_id")
        _id(self.camera_id, "camera", "camera_id")
        if self.trigger_type not in {"motion", "scene_change"}:
            raise SemanticContractError("unknown trigger_type")
        for name in ("trigger_started_at", "trigger_ended_at", "clip_start", "clip_end"):
            _time(getattr(self, name), name)
        if not (
            datetime.fromisoformat(self.clip_start)
            <= datetime.fromisoformat(self.trigger_started_at)
            <= datetime.fromisoformat(self.trigger_ended_at)
            <= datetime.fromisoformat(self.clip_end)
        ):
            raise SemanticContractError("candidate timestamps are out of order")
        _positive_text(self.detector, "detector")
        if not isinstance(self.detector_config, Mapping) or not self.detector_config:
            raise SemanticContractError("detector_config must be a nonempty object")
        for key, value in self.detector_config.items():
            if not isinstance(key, str) or not key or (
                type(value) not in (bool, int, float)
                or (isinstance(value, float) and not math.isfinite(value))
            ):
                raise SemanticContractError("detector_config must contain finite scalar settings")
        object.__setattr__(self, "detector_config", MappingProxyType(dict(self.detector_config)))
        _confidence(self.peak_score, "peak_score")
        _confidence(self.average_score, "average_score")
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise SemanticContractError("sample_count must be positive")
        if (type(self.sequence_start) is not int or type(self.sequence_end) is not int
                or self.sequence_start < 1 or self.sequence_end < self.sequence_start):
            raise SemanticContractError("sequence range is invalid")
        if not isinstance(self.media_sha256, str) or not _HASH.fullmatch(self.media_sha256):
            raise SemanticContractError("media_sha256 must be lowercase SHA-256")

    @classmethod
    def from_stage2(cls, record: Mapping[str, Any]) -> VisualEventCandidate:
        """Copy only stable source facts; never trust semantic data from the detector."""
        try:
            return cls(**{name: record[name] for name in cls.__dataclass_fields__})
        except KeyError as error:
            raise SemanticContractError(f"Stage 2 candidate lacks {error.args[0]}") from error


@dataclass(frozen=True)
class AnalysisProvenance:
    model_id: str
    model_version: str

    def __post_init__(self) -> None:
        _positive_text(self.model_id, "model_id")
        _positive_text(self.model_version, "model_version")


@dataclass(frozen=True)
class CategoryObservation:
    label: str
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or self.label not in CATEGORY_LABELS:
            raise SemanticContractError("unknown category label")
        _confidence(self.confidence, "category confidence")


@dataclass(frozen=True)
class ActivityObservation:
    label: str
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or self.label not in ACTIVITY_LABELS:
            raise SemanticContractError("unknown activity label")
        _confidence(self.confidence, "activity confidence")


@dataclass(frozen=True)
class SemanticFingerprint:
    """Comparable symbolic signature; no physical identity or vector store implied."""

    categories: tuple[str, ...]
    activity: str | None
    opaque_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.categories, tuple) or any(
            not isinstance(label, str) for label in self.categories
        ):
            raise SemanticContractError("fingerprint categories must be a tuple of labels")
        if tuple(sorted(set(self.categories))) != self.categories:
            raise SemanticContractError("fingerprint categories must be sorted and unique")
        if any(label not in CATEGORY_LABELS for label in self.categories):
            raise SemanticContractError("unknown fingerprint category")
        if self.activity is not None and (
            not isinstance(self.activity, str) or self.activity not in ACTIVITY_LABELS
        ):
            raise SemanticContractError("unknown fingerprint activity")
        if self.opaque_hash is not None and (
            not isinstance(self.opaque_hash, str) or not _HASH.fullmatch(self.opaque_hash)
        ):
            raise SemanticContractError("opaque_hash must be lowercase SHA-256")


def fingerprint_for(
    observations: tuple[CategoryObservation, ...],
    activity: ActivityObservation | None,
) -> SemanticFingerprint:
    return SemanticFingerprint(tuple(sorted({item.label for item in observations})),
                               None if activity is None else activity.label)


@dataclass(frozen=True)
class PromotionRecommendation:
    """Advisory only: creating this value neither uploads nor deletes evidence."""

    decision: PromotionDecision
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.decision, PromotionDecision):
            raise SemanticContractError("unknown promotion decision")
        _code(self.reason, "promotion reason")


@dataclass(frozen=True)
class SemanticCandidate:
    source: VisualEventCandidate
    analyzed_at: str
    semantic_status: SemanticStatus
    analysis: AnalysisProvenance | None = None
    observations: tuple[CategoryObservation, ...] = ()
    activity: ActivityObservation | None = None
    semantic_fingerprint: SemanticFingerprint | None = None
    promotion: PromotionRecommendation | None = None
    failure_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, VisualEventCandidate):
            raise SemanticContractError("source must be a Stage 2 visual event candidate")
        _time(self.analyzed_at, "analyzed_at")
        if datetime.fromisoformat(self.analyzed_at) < datetime.fromisoformat(self.source.clip_end):
            raise SemanticContractError("analysis cannot precede the evidence clip")
        if not isinstance(self.semantic_status, SemanticStatus):
            raise SemanticContractError("unknown semantic_status")
        if self.analysis is not None and not isinstance(self.analysis, AnalysisProvenance):
            raise SemanticContractError("analysis must identify a model")
        if not isinstance(self.observations, tuple) or any(
            not isinstance(item, CategoryObservation) for item in self.observations
        ):
            raise SemanticContractError("observations must be category observations")
        if len({item.label for item in self.observations}) != len(self.observations):
            raise SemanticContractError("duplicate category observations")
        if self.activity is not None and not isinstance(self.activity, ActivityObservation):
            raise SemanticContractError("activity must be a coarse activity observation")
        if self.promotion is not None and not isinstance(self.promotion, PromotionRecommendation):
            raise SemanticContractError("promotion must be an advisory decision")
        if self.semantic_status == SemanticStatus.SUCCESS:
            if self.analysis is None or self.failure_code is not None:
                raise SemanticContractError("success requires analysis and forbids failure_code")
            expected = fingerprint_for(self.observations, self.activity)
            if not isinstance(self.semantic_fingerprint, SemanticFingerprint) or (
                self.semantic_fingerprint.categories != expected.categories
                or self.semantic_fingerprint.activity != expected.activity
            ):
                raise SemanticContractError("fingerprint must match semantic observations")
        else:
            if self.failure_code is None:
                raise SemanticContractError("non-success requires failure_code")
            _code(self.failure_code, "failure_code")
            if self.observations or self.activity or self.semantic_fingerprint or self.promotion:
                raise SemanticContractError("failed analysis cannot claim semantics or promotion")

    @property
    def candidate_id(self) -> str:
        return self.source.candidate_id

    @property
    def evidence_id(self) -> str:
        return self.source.evidence_id

    @property
    def embodiment_id(self) -> str:
        return self.source.embodiment_id


def semantic_candidate_as_mapping(value: SemanticCandidate) -> dict[str, Any]:
    """Stable JSON-ready V1 representation. Confidence is semantic, never p95."""
    if not isinstance(value, SemanticCandidate):
        raise SemanticContractError("expected SemanticCandidate")
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            name: (dict(value.source.detector_config) if name == "detector_config"
                   else getattr(value.source, name))
            for name in VisualEventCandidate.__dataclass_fields__
        },
        "analyzed_at": value.analyzed_at,
        "semantic_status": value.semantic_status.value,
        "analysis": None if value.analysis is None else {
            "model_id": value.analysis.model_id, "model_version": value.analysis.model_version,
        },
        "observations": [
            {"label": item.label, "confidence": item.confidence} for item in value.observations
        ],
        "activity": None if value.activity is None else {
            "label": value.activity.label, "confidence": value.activity.confidence,
        },
        "semantic_fingerprint": None if value.semantic_fingerprint is None else {
            "categories": list(value.semantic_fingerprint.categories),
            "activity": value.semantic_fingerprint.activity,
            "opaque_hash": value.semantic_fingerprint.opaque_hash,
        },
        "promotion": None if value.promotion is None else {
            "decision": value.promotion.decision.value,
            "reason": value.promotion.reason,
        },
        "failure_code": value.failure_code,
    }


def parse_semantic_candidate(value: Mapping[str, Any]) -> SemanticCandidate:
    """Reject unknown V1 fields so authority-bearing additions cannot be ignored."""
    item = _mapping(value, "semantic candidate")
    allowed = {
        "schema_version", "source", "analyzed_at", "semantic_status", "analysis",
        "observations", "activity", "semantic_fingerprint", "promotion", "failure_code",
    }
    if set(item) - allowed:
        raise SemanticContractError("unknown semantic candidate fields")
    version = item.get("schema_version", SCHEMA_VERSION)
    if type(version) is not int or version != SCHEMA_VERSION:
        raise SemanticContractError("unsupported semantic candidate schema version")
    try:
        source = _mapping(item["source"], "source")
        if set(source) != set(VisualEventCandidate.__dataclass_fields__):
            raise SemanticContractError("source fields do not match Stage 2 contract")
        analysis = item.get("analysis")
        activity = item.get("activity")
        fingerprint = item.get("semantic_fingerprint")
        promotion = item.get("promotion")
        observations = item.get("observations", [])
        if not isinstance(observations, list):
            raise SemanticContractError("observations must be a list")
        return SemanticCandidate(
            source=VisualEventCandidate(**source),
            analyzed_at=item["analyzed_at"],
            semantic_status=SemanticStatus(item["semantic_status"]),
            analysis=None if analysis is None else AnalysisProvenance(**_mapping(analysis, "analysis")),
            observations=tuple(CategoryObservation(**_mapping(row, "observation")) for row in observations),
            activity=None if activity is None else ActivityObservation(**_mapping(activity, "activity")),
            semantic_fingerprint=None if fingerprint is None else SemanticFingerprint(
                categories=tuple(_mapping(fingerprint, "semantic_fingerprint")["categories"]),
                activity=fingerprint["activity"],
                opaque_hash=fingerprint.get("opaque_hash"),
            ),
            promotion=None if promotion is None else PromotionRecommendation(
                decision=PromotionDecision(_mapping(promotion, "promotion")["decision"]),
                reason=promotion["reason"],
            ),
            failure_code=item.get("failure_code"),
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, SemanticContractError):
            raise
        raise SemanticContractError(f"invalid semantic candidate: {error}") from error
