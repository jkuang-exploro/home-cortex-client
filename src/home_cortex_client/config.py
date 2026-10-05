"""Device-local configuration for the edge runtime."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

ENV_PREFIX = "HOME_CORTEX_CLIENT_"


@dataclass(frozen=True)
class ChangeDetectionConfig:
    """Thresholds for local visual-change detection and candidate clips.

    Scores are mean absolute grayscale difference divided by 255, so 0 is an
    unchanged sample and 1 is a complete inversion. A candidate clip is the
    trigger interval plus pre-roll and post-roll. ``maximum_candidate_duration_s``
    caps the trigger interval. When motion continues past that cap, the open
    interval is closed and a new one starts. Those pieces are not merged back
    together. Pre-roll and post-roll make neighboring clips overlap.

    ``cooldown_s`` suppresses a new interval that starts in that many seconds
    after a candidate is packaged. The default is 0. ``merge_gap_s`` is what
    joins pauses. Candidates stay on this machine.
    """

    sample_hz: float = 4.0
    motion_start_threshold: float = 0.08
    motion_continue_threshold: float = 0.04
    scene_change_threshold: float = 0.45
    settling_s: float = 1.0
    minimum_event_duration_s: float = 0.4
    pre_roll_s: float = 3.0
    post_roll_s: float = 5.0
    merge_gap_s: float = 2.0
    minimum_peak_score: float = 0.08
    maximum_candidate_duration_s: float = 30.0
    cooldown_s: float = 0.0
    max_candidates: int = 32
    max_candidate_age_s: float = 6 * 3600.0
    max_candidate_disk_bytes: int = 256 * 1024 * 1024
    scene_change_candidates: bool = True
    grid_width: int = 32
    grid_height: int = 24

    def __post_init__(self) -> None:
        _unit("sample_hz", self.sample_hz, upper=1000)
        _unit("motion_continue_threshold", self.motion_continue_threshold)
        _unit("motion_start_threshold", self.motion_start_threshold)
        _unit("scene_change_threshold", self.scene_change_threshold)
        if not (
            self.motion_continue_threshold
            <= self.motion_start_threshold
            <= self.scene_change_threshold
        ):
            raise ValueError(
                "thresholds must be continue <= start <= scene change, each in (0, 1]"
            )
        _unit("minimum_peak_score", self.minimum_peak_score, allow_zero=True)
        for name in (
            "settling_s", "minimum_event_duration_s", "pre_roll_s", "post_roll_s",
            "merge_gap_s", "cooldown_s",
        ):
            _duration(name, getattr(self, name), allow_zero=True)
        _duration("maximum_candidate_duration_s", self.maximum_candidate_duration_s)
        _duration("max_candidate_age_s", self.max_candidate_age_s)
        if type(self.max_candidates) is not int or self.max_candidates < 1:
            raise ValueError("max_candidates must be a positive integer")
        if type(self.max_candidate_disk_bytes) is not int or self.max_candidate_disk_bytes < 1:
            raise ValueError("max_candidate_disk_bytes must be a positive integer")
        if type(self.grid_width) is not int or type(self.grid_height) is not int:
            raise ValueError("detector grid dimensions must be positive integers")
        if self.grid_width < 1 or self.grid_height < 1:
            raise ValueError("detector grid dimensions must be positive integers")
        if not isinstance(self.scene_change_candidates, bool):
            raise ValueError("scene_change_candidates must be true or false")  # noqa: TRY004 — preserve configuration error contract

    @property
    def sample_interval_s(self) -> float:
        return 1.0 / self.sample_hz


@dataclass(frozen=True)
class ClientConfig:
    source: str = "mac"
    embodiment_id: str = "embodiment:macbook-0"
    cortex_url: str | None = None
    cortex_api_key: str | None = None
    client_interface: str = "v1"
    credentials_dir: str | None = None
    state_dir: str | None = None
    device_id: str = "device:dev_macbook"
    camera_id: str = "camera:built_in"
    camera_index: int = 0
    stream_host: str = "127.0.0.1"
    stream_port: int = 8088
    width: int | None = None
    height: int | None = None
    fps: float = 10.0
    buffer_seconds: float = 60.0
    freshness_seconds: float = 2.0
    evidence_dir: str | None = None
    evidence_max_items: int = 8
    detection: ChangeDetectionConfig = field(default_factory=ChangeDetectionConfig)
    analyzer: str = "off"
    promotion_window_s: float = 8.0
    promotion_min_confidence: float = 0.5
    promotion_repeat: str = "retain"
    promotion_override: str | None = None
    session_capabilities: tuple[str, ...] = ("vision.observe",)
    promotion_queue_limit: int = 8
    promotion_max_age_s: float = 6 * 3600.0

    def __post_init__(self) -> None:
        if self.client_interface not in {"v1", "legacy"}:
            raise ValueError("client_interface must be v1 or explicit legacy rollback")
        if self.analyzer not in {"off", "vision", "hog"}:
            raise ValueError("analyzer must be off, vision, or hog")
        _duration("promotion_window_s", self.promotion_window_s, allow_zero=True)
        _unit("promotion_min_confidence", self.promotion_min_confidence, allow_zero=True)
        if self.promotion_repeat not in {"retain", "drop"}:
            raise ValueError("promotion_repeat must be retain or drop")
        if self.promotion_override not in {None, "force_promote", "force_retain"}:
            raise ValueError("promotion_override must be force_promote, force_retain, or unset")
        allowed = ("vision.observe", "vision.observe_clip", "vision.autonomous_promotion")
        if any(name not in allowed for name in self.session_capabilities):
            raise ValueError(
                "session_capabilities must be vision.observe, vision.observe_clip, "
                "or vision.autonomous_promotion"
            )
        if type(self.promotion_queue_limit) is not int or self.promotion_queue_limit < 1:
            raise ValueError("promotion_queue_limit must be a positive integer")
        _duration("promotion_max_age_s", self.promotion_max_age_s)

    @classmethod
    def from_env(cls) -> ClientConfig:
        detection = ChangeDetectionConfig(
            sample_hz=_number("DETECTOR_HZ", ChangeDetectionConfig.sample_hz),
            motion_start_threshold=_number(
                "MOTION_START_THRESHOLD", ChangeDetectionConfig.motion_start_threshold,
            ),
            motion_continue_threshold=_number(
                "MOTION_CONTINUE_THRESHOLD", ChangeDetectionConfig.motion_continue_threshold,
            ),
            scene_change_threshold=_number(
                "SCENE_CHANGE_THRESHOLD", ChangeDetectionConfig.scene_change_threshold,
            ),
            settling_s=_number("SETTLING_SECONDS", ChangeDetectionConfig.settling_s),
            minimum_event_duration_s=_number(
                "MINIMUM_EVENT_DURATION", ChangeDetectionConfig.minimum_event_duration_s,
            ),
            pre_roll_s=_number("PRE_ROLL_SECONDS", ChangeDetectionConfig.pre_roll_s),
            post_roll_s=_number("POST_ROLL_SECONDS", ChangeDetectionConfig.post_roll_s),
            merge_gap_s=_number("MERGE_GAP_SECONDS", ChangeDetectionConfig.merge_gap_s),
            minimum_peak_score=_number(
                "MINIMUM_PEAK_SCORE", ChangeDetectionConfig.minimum_peak_score,
            ),
            maximum_candidate_duration_s=_number(
                "MAXIMUM_CANDIDATE_DURATION",
                ChangeDetectionConfig.maximum_candidate_duration_s,
            ),
            cooldown_s=_number("COOLDOWN_SECONDS", ChangeDetectionConfig.cooldown_s),
            max_candidates=_integer("MAX_CANDIDATES", ChangeDetectionConfig.max_candidates),
            max_candidate_age_s=_number(
                "MAX_CANDIDATE_AGE_SECONDS", ChangeDetectionConfig.max_candidate_age_s,
            ),
            max_candidate_disk_bytes=_integer(
                "MAX_CANDIDATE_DISK_BYTES", ChangeDetectionConfig.max_candidate_disk_bytes,
            ),
            scene_change_candidates=_bool(
                "SCENE_CHANGE_CANDIDATES", ChangeDetectionConfig.scene_change_candidates,
            ),
        )
        return cls(
            source=_text("SOURCE", cls.source),
            embodiment_id=_text("EMBODIMENT_ID", cls.embodiment_id),
            cortex_url=_value("CORTEX_URL"),
            cortex_api_key=_value("CORTEX_API_KEY"),
            client_interface=_text("INTERFACE", cls.client_interface),
            credentials_dir=_value("CREDENTIALS_DIR"),
            state_dir=_value("STATE_DIR"),
            device_id=_text("DEVICE_ID", cls.device_id),
            camera_id=_text("CAMERA_ID", cls.camera_id),
            camera_index=_integer("CAMERA_INDEX", cls.camera_index),
            stream_host=_text("STREAM_HOST", cls.stream_host),
            stream_port=_integer("STREAM_PORT", cls.stream_port),
            width=_optional_integer("WIDTH"),
            height=_optional_integer("HEIGHT"),
            fps=_number("FPS", cls.fps),
            buffer_seconds=_number("BUFFER_SECONDS", cls.buffer_seconds),
            freshness_seconds=_number("FRESHNESS_SECONDS", cls.freshness_seconds),
            evidence_dir=_value("EVIDENCE_DIR"),
            evidence_max_items=_integer("EVIDENCE_MAX_ITEMS", cls.evidence_max_items),
            detection=detection,
            analyzer=_analyzer_name(),
            promotion_window_s=_number("PROMOTION_WINDOW_SECONDS", cls.promotion_window_s),
            promotion_min_confidence=_number(
                "PROMOTION_MIN_CONFIDENCE", cls.promotion_min_confidence,
            ),
            promotion_repeat=_promotion_repeat(),
            promotion_override=_promotion_override(),
            session_capabilities=_session_capabilities(),
            promotion_queue_limit=_integer("PROMOTION_QUEUE_LIMIT", cls.promotion_queue_limit),
            promotion_max_age_s=_number("PROMOTION_MAX_AGE_SECONDS", cls.promotion_max_age_s),
        )


def _value(name: str) -> str | None:
    value = os.environ.get(f"{ENV_PREFIX}{name}")
    if value is None or not value.strip():
        return None
    return value.strip()


def _text(name: str, default: str) -> str:
    return _value(name) or default


def _integer(name: str, default: int) -> int:
    value = _value(name)
    return default if value is None else int(value)


def _optional_integer(name: str) -> int | None:
    value = _value(name)
    return None if value is None else int(value)


def _number(name: str, default: float) -> float:
    value = _value(name)
    return default if value is None else float(value)


def _analyzer_name() -> str:
    value = (_value("ANALYZER") or "off").lower()
    if value not in {"off", "vision", "hog"}:
        raise ValueError("HOME_CORTEX_CLIENT_ANALYZER must be off, vision, or hog")
    return value


def _promotion_repeat() -> str:
    value = (_value("PROMOTION_REPEAT") or "retain").lower()
    if value not in {"retain", "drop"}:
        raise ValueError("HOME_CORTEX_CLIENT_PROMOTION_REPEAT must be retain or drop")
    return value


def _session_capabilities() -> tuple[str, ...]:
    value = _value("CAPABILITIES")
    if value is None:
        return ("vision.observe",)
    requested = [part.strip() for part in value.split(",") if part.strip()]
    allowed = ("vision.observe", "vision.observe_clip", "vision.autonomous_promotion")
    if not requested or any(name not in allowed for name in requested):
        raise ValueError(
            "HOME_CORTEX_CLIENT_CAPABILITIES must list vision.observe, "
            "vision.observe_clip, or vision.autonomous_promotion"
        )
    return tuple(name for name in allowed if name in requested)


def _promotion_override() -> str | None:
    value = _value("PROMOTION_OVERRIDE")
    if value is None:
        return None
    lowered = value.lower()
    if lowered not in {"force_promote", "force_retain"}:
        raise ValueError(
            "HOME_CORTEX_CLIENT_PROMOTION_OVERRIDE must be force_promote or force_retain"
        )
    return lowered


def _bool(name: str, default: bool) -> bool:
    value = _value(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _unit(name: str, value: float, *, allow_zero: bool = False, upper: float = 1.0) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value > upper or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} is outside its allowed range")


def _duration(name: str, value: float, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite duration")
    if value < 0 or (not allow_zero and value <= 0):
        raise ValueError(f"{name} must be a positive duration")
