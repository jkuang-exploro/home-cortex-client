"""Bounded local history of encoded camera segments."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock


class BufferError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class CapturedSegment:
    """One short encoded unit. ``captured_at`` is the capture clock, not upload time."""

    camera_id: str
    sequence_number: int
    captured_at: datetime
    duration_s: float
    width: int
    height: int
    codec: str
    content_type: str
    local_id: str
    payload: bytes

    def ends_at(self) -> datetime:
        return self.captured_at + timedelta(seconds=self.duration_s)

    def as_mapping(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "sequence_number": self.sequence_number,
            "captured_at": self.captured_at.isoformat(timespec="milliseconds"),
            "duration": self.duration_s,
            "resolution": {"width": self.width, "height": self.height},
            "codec": self.codec,
            "format": self.content_type,
            "local_id": self.local_id,
            "byte_length": len(self.payload),
        }


class RingBuffer:
    """Keep recent encoded segments only. Sequence numbers survive eviction."""

    def __init__(self, *, duration_s: float = 60.0, max_bytes: int = 256 * 1024 * 1024) -> None:
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise BufferError("invalid_duration", "buffer duration must be positive")
        if max_bytes < 1:
            raise BufferError("invalid_duration", "buffer byte cap must be positive")
        self.duration_s = float(duration_s)
        self.max_bytes = int(max_bytes)
        self._segments: deque[CapturedSegment] = deque()
        self._sequence = 0
        self._bytes = 0
        self._lock = Lock()

    def append_frame(
        self,
        *,
        camera_id: str,
        captured_at: datetime,
        width: int,
        height: int,
        payload: bytes,
        duration_s: float,
    ) -> CapturedSegment:
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise BufferError("capture_failure", "camera_id is required")
        if captured_at.tzinfo is None:
            raise BufferError("capture_failure", "captured_at must be timezone-aware")
        if not isinstance(payload, (bytes, bytearray)) or not payload:
            raise BufferError("capture_failure", "encoded payload is required")
        if type(width) is not int or type(height) is not int or width < 1 or height < 1:
            raise BufferError("capture_failure", "resolution must be positive integers")
        if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
            raise BufferError("invalid_duration", "frame duration must be positive")
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise BufferError("invalid_duration", "frame duration must be positive")
        encoded = bytes(payload)
        with self._lock:
            if self._segments and captured_at < self._segments[-1].captured_at:
                raise BufferError(
                    "timestamp_regressed",
                    "capture timestamps must not move backwards",
                )
            self._sequence += 1
            segment = CapturedSegment(
                camera_id=camera_id,
                sequence_number=self._sequence,
                captured_at=captured_at,
                duration_s=float(duration_s),
                width=width,
                height=height,
                codec="jpeg",
                content_type="image/jpeg",
                local_id=f"local:{self._sequence}",
                payload=encoded,
            )
            self._segments.append(segment)
            self._bytes += len(encoded)
            self._evict()
            return segment

    def latest(self) -> CapturedSegment | None:
        with self._lock:
            return self._segments[-1] if self._segments else None

    def latest_sequence(self) -> int:
        with self._lock:
            return self._segments[-1].sequence_number if self._segments else 0

    def segments(self) -> tuple[CapturedSegment, ...]:
        with self._lock:
            return tuple(self._segments)

    def span(self) -> tuple[str | None, str | None, float]:
        """Return buffer start, end, and retained duration in seconds."""
        with self._lock:
            if not self._segments:
                return None, None, 0.0
            start = self._segments[0].captured_at
            end = self._segments[-1].ends_at()
            return (
                start.isoformat(timespec="milliseconds"),
                end.isoformat(timespec="milliseconds"),
                round((end - start).total_seconds(), 3),
            )

    def select_recent(self, seconds: float, *, now: datetime) -> tuple[CapturedSegment, ...]:
        """Select segments overlapping ``[now - seconds, now]``.

        The returned range is the captured span, which can differ from the
        requested window by segment boundaries. A window the buffer never
        recorded raises ``buffer_too_short``.
        """
        requested = _positive_seconds(seconds)
        if requested > self.duration_s:
            raise BufferError(
                "invalid_duration",
                "requested duration is longer than the local buffer",
            )
        if now.tzinfo is None:
            raise BufferError("invalid_duration", "selection time must be timezone-aware")
        with self._lock:
            retained = list(self._segments)
        if not retained:
            raise BufferError("buffer_too_short", "buffer has no captured video")
        window_start = now - timedelta(seconds=requested)
        chosen = [
            segment for segment in retained
            if segment.captured_at <= now and segment.ends_at() > window_start
        ]
        if not chosen:
            raise BufferError("buffer_too_short", "buffer does not cover the requested interval")
        slack = max(chosen[0].duration_s * 2, 0.5)
        if chosen[0].captured_at > window_start + timedelta(seconds=slack):
            raise BufferError("buffer_too_short", "buffer does not cover the requested interval")
        return tuple(chosen)

    def overlapping(self, start: datetime, end: datetime) -> tuple[CapturedSegment, ...]:
        """Return segments that overlap ``[start, end]``, using actual capture times."""
        if start.tzinfo is None or end.tzinfo is None:
            raise BufferError("invalid_duration", "selection time must be timezone-aware")
        if end < start:
            raise BufferError("invalid_duration", "selection end is before the start")
        with self._lock:
            retained = list(self._segments)
        return tuple(
            segment for segment in retained
            if segment.captured_at <= end and segment.ends_at() > start
        )

    def _evict(self) -> None:
        while self._segments:
            newest = self._segments[-1].captured_at
            oldest = self._segments[0]
            expired = oldest.captured_at < newest - timedelta(seconds=self.duration_s)
            over_budget = self._bytes > self.max_bytes and len(self._segments) > 1
            if not expired and not over_budget:
                return
            removed = self._segments.popleft()
            self._bytes -= len(removed.payload)


def _positive_seconds(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BufferError("invalid_duration", "duration must be a positive number of seconds")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise BufferError("invalid_duration", "duration must be a positive number of seconds")
    return float(value)
