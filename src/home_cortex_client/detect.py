"""Deterministic visual-change detector.

Compares consecutive downsampled grayscale samples. It does not name objects
or people. A motion interval stays open while the score remains above the
continue threshold, and it closes after the settling period. An interval that
reaches ``maximum_candidate_duration_s`` closes with ``forced_split`` so a
later stage can keep the next interval separate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import ChangeDetectionConfig


DETECTOR_NAME = "frame_difference"


@dataclass(frozen=True)
class MotionEvent:
    started_at: datetime
    ended_at: datetime
    peak_score: float
    average_score: float
    sample_count: int
    forced_split: bool = False
    detector: str = DETECTOR_NAME

    def as_mapping(self) -> dict[str, object]:
        return {
            "type": "motion",
            "started_at": _iso(self.started_at),
            "ended_at": _iso(self.ended_at),
            "peak_score": self.peak_score,
            "average_score": self.average_score,
            "detector": self.detector,
            "forced_split": self.forced_split,
        }


@dataclass(frozen=True)
class SceneChangeEvent:
    occurred_at: datetime
    score: float
    detector: str = DETECTOR_NAME

    def as_mapping(self) -> dict[str, object]:
        return {
            "type": "scene_change",
            "occurred_at": _iso(self.occurred_at),
            "score": self.score,
            "detector": self.detector,
        }


class ChangeDetector:
    """Score sampled frames and emit motion and scene-change boundaries."""

    def __init__(self, config: ChangeDetectionConfig | None = None) -> None:
        self.config = config or ChangeDetectionConfig()
        self.state = "idle"
        self.last_score = 0.0
        self.frames_sampled = 0
        self.scored_samples = 0
        self.motion_events = 0
        self.scene_changes = 0
        self.suppressed_short = 0
        self._previous: bytes | None = None
        self._last_sample_at: datetime | None = None
        self._started_at: datetime | None = None
        self._last_active_at: datetime | None = None
        self._peak = 0.0
        self._score_sum = 0.0
        self._score_count = 0

    def observe(
        self, when: datetime, sample: bytes, *, width: int, height: int,
    ) -> list[MotionEvent | SceneChangeEvent]:
        if when.tzinfo is None:
            raise ValueError("detector time must be timezone-aware")
        if not isinstance(sample, (bytes, bytearray)) or width * height != len(sample):
            raise ValueError("grayscale sample size must match width and height")
        if self._last_sample_at is not None:
            elapsed = (when - self._last_sample_at).total_seconds()
            if elapsed < self.config.sample_interval_s - 1e-9:
                return self._settled(when)
        gray = fit_grid(bytes(sample), width, height, self.config.grid_width, self.config.grid_height)
        self.frames_sampled += 1
        signals: list[MotionEvent | SceneChangeEvent] = []
        if self._previous is None:
            self._previous = gray
            self._last_sample_at = when
            return signals
        score = change_score(self._previous, gray)
        self._previous = gray
        self._last_sample_at = when
        self.last_score = score
        self.scored_samples += 1
        if score >= self.config.scene_change_threshold:
            self.scene_changes += 1
            signals.append(SceneChangeEvent(when, score))
        signals.extend(self._advance_motion(when, score))
        return signals

    def poll(self, when: datetime) -> list[MotionEvent | SceneChangeEvent]:
        if when.tzinfo is None:
            raise ValueError("detector time must be timezone-aware")
        return self._settled(when)

    def active_interval(self) -> tuple[datetime, datetime] | None:
        """Return the open motion interval, or nothing while idle."""
        if self.state != "active" or self._started_at is None or self._last_active_at is None:
            return None
        return self._started_at, self._last_active_at

    def status(self) -> dict[str, object]:
        config = self.config
        return {
            "state": self.state,
            "detector": DETECTOR_NAME,
            "last_score": self.last_score,
            "frames_sampled": self.frames_sampled,
            "scored_samples": self.scored_samples,
            "motion_events": self.motion_events,
            "scene_changes": self.scene_changes,
            "suppressed_short": self.suppressed_short,
            "started_at": None if self._started_at is None or self.state != "active" else _iso(self._started_at),
            "last_active_at": None if self._last_active_at is None or self.state != "active" else _iso(self._last_active_at),
            "peak_score": round(self._peak, 6) if self.state == "active" else None,
            "average_score": (
                round(self._score_sum / self._score_count, 6)
                if self.state == "active" and self._score_count else None
            ),
            "sample_hz": config.sample_hz,
            "motion_start_threshold": config.motion_start_threshold,
            "motion_continue_threshold": config.motion_continue_threshold,
            "scene_change_threshold": config.scene_change_threshold,
            "settling_s": config.settling_s,
            "minimum_event_duration_s": config.minimum_event_duration_s,
            "maximum_candidate_duration_s": config.maximum_candidate_duration_s,
        }

    def _advance_motion(self, when: datetime, score: float) -> list[MotionEvent]:
        if self.state == "active" and self._started_at is not None:
            elapsed = (when - self._started_at).total_seconds()
            if elapsed >= self.config.maximum_candidate_duration_s:
                ended = self._started_at + timedelta(seconds=self.config.maximum_candidate_duration_s)
                closed = self._finish(ended, forced_split=True)
                signals = [closed] if closed is not None else []
                if score >= self.config.motion_continue_threshold:
                    self._begin(ended, score)
                    if when > ended:
                        self._last_active_at = when
                return signals
        if self.state == "idle":
            if score >= self.config.motion_start_threshold:
                self._begin(when, score)
            return []
        if score >= self.config.motion_continue_threshold:
            self._mark_active(when, score)
            return []
        return self._settled(when)

    def _settled(self, when: datetime) -> list[MotionEvent]:
        if self.state != "active" or self._last_active_at is None:
            return []
        quiet = (when - self._last_active_at).total_seconds()
        if quiet < self.config.settling_s:
            return []
        closed = self._finish(self._last_active_at, forced_split=False)
        return [closed] if closed is not None else []

    def _begin(self, when: datetime, score: float) -> None:
        self.state = "active"
        self._started_at = when
        self._last_active_at = when
        self._peak = score
        self._score_sum = score
        self._score_count = 1

    def _mark_active(self, when: datetime, score: float) -> None:
        self._last_active_at = when
        self._peak = max(self._peak, score)
        self._score_sum += score
        self._score_count += 1

    def _finish(self, ended_at: datetime, *, forced_split: bool) -> MotionEvent | None:
        started = self._started_at
        count = self._score_count
        peak = self._peak
        average = self._score_sum / count if count else 0.0
        self.state = "idle"
        self._started_at = None
        self._last_active_at = None
        self._peak = 0.0
        self._score_sum = 0.0
        self._score_count = 0
        if started is None or count < 1:
            return None
        duration = (ended_at - started).total_seconds()
        if not forced_split and duration < self.config.minimum_event_duration_s:
            self.suppressed_short += 1
            return None
        self.motion_events += 1
        return MotionEvent(
            started_at=started,
            ended_at=ended_at,
            peak_score=round(peak, 6),
            average_score=round(average, 6),
            sample_count=count,
            forced_split=forced_split,
        )


def change_score(previous: bytes, current: bytes) -> float:
    """Return the normalized mean absolute difference of two equal grids."""
    if len(previous) != len(current) or not previous:
        raise ValueError("detector frames must be equal and non-empty")
    total = 0
    for left, right in zip(previous, current):
        total += abs(left - right)
    return total / (len(previous) * 255.0)


def fit_grid(sample: bytes, width: int, height: int, grid_width: int, grid_height: int) -> bytes:
    """Box-filter a row-major grayscale sample onto the detector grid."""
    if width == grid_width and height == grid_height:
        return sample
    output = bytearray(grid_width * grid_height)
    for gy in range(grid_height):
        y0 = gy * height // grid_height
        y1 = max((gy + 1) * height // grid_height, y0 + 1)
        for gx in range(grid_width):
            x0 = gx * width // grid_width
            x1 = max((gx + 1) * width // grid_width, x0 + 1)
            total = 0
            count = 0
            for y in range(y0, y1):
                row = y * width
                for x in range(x0, x1):
                    total += sample[row + x]
                    count += 1
            output[gy * grid_width + gx] = total // count
    return bytes(output)


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds")
