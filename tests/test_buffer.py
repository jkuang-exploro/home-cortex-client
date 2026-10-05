"""Ring buffer ordering, eviction, and explicit selection. No camera required."""
from datetime import datetime, timedelta, timezone

import pytest

from home_cortex_client.buffer import BufferError, RingBuffer
from home_cortex_client.sources import TINY_JPEG


START = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def _append(buffer: RingBuffer, offset_s: float, *, payload: bytes = TINY_JPEG, duration: float = 0.1):
    return buffer.append_frame(
        camera_id="camera:built_in",
        captured_at=START + timedelta(seconds=offset_s),
        width=8,
        height=4,
        payload=payload,
        duration_s=duration,
    )


def test_overlapping_uses_segment_bounds() -> None:
    buffer = RingBuffer(duration_s=60)
    _append(buffer, 0, duration=1)
    _append(buffer, 1, duration=1)
    point = buffer.overlapping(START + timedelta(seconds=0.5), START + timedelta(seconds=0.5))
    assert [segment.sequence_number for segment in point] == [1]
    both = buffer.overlapping(START, START + timedelta(seconds=1))
    assert [segment.sequence_number for segment in both] == [1, 2]
    with pytest.raises(BufferError) as error:
        buffer.overlapping(START + timedelta(seconds=2), START)
    assert error.value.code == "invalid_duration"


def test_sequence_and_timestamps_increase_and_latest_is_newest() -> None:
    buffer = RingBuffer(duration_s=60)
    first = _append(buffer, 0)
    second = _append(buffer, 0.1)
    assert first.sequence_number == 1
    assert second.sequence_number == 2
    assert second.local_id == "local:2"
    assert first.captured_at < second.captured_at
    assert buffer.latest() == second
    assert [segment.sequence_number for segment in buffer.segments()] == [1, 2]


def test_timestamp_regression_is_rejected_without_consuming_the_sequence() -> None:
    buffer = RingBuffer(duration_s=60)
    _append(buffer, 1)
    with pytest.raises(BufferError) as error:
        _append(buffer, 0.5)
    assert error.value.code == "timestamp_regressed"
    assert buffer.latest_sequence() == 1
    _append(buffer, 1.2)
    assert [segment.sequence_number for segment in buffer.segments()] == [1, 2]


def test_oldest_segments_are_evicted_by_time_and_bytes() -> None:
    timed = RingBuffer(duration_s=1.0)
    _append(timed, 0.0)
    _append(timed, 0.5)
    _append(timed, 1.2)
    retained = [segment.captured_at for segment in timed.segments()]
    assert retained[0] == START + timedelta(seconds=0.5)
    assert retained[-1] == START + timedelta(seconds=1.2)
    assert timed.latest_sequence() == 3

    bounded = RingBuffer(duration_s=60, max_bytes=len(TINY_JPEG) * 2)
    for offset in (0, 1, 2):
        _append(bounded, offset, payload=TINY_JPEG)
    assert len(bounded.segments()) == 2
    assert [segment.sequence_number for segment in bounded.segments()] == [2, 3]
    assert sum(len(segment.payload) for segment in bounded.segments()) <= len(TINY_JPEG) * 2


def test_select_recent_reports_actual_bounds_and_rejects_a_short_buffer() -> None:
    buffer = RingBuffer(duration_s=60)
    for offset in (0, 1, 2, 3, 4, 5):
        _append(buffer, offset, duration=1)
    chosen = buffer.select_recent(3, now=START + timedelta(seconds=5))
    assert chosen[0].captured_at == START + timedelta(seconds=2)
    assert chosen[-1].ends_at() == START + timedelta(seconds=6)
    assert [segment.sequence_number for segment in chosen] == [3, 4, 5, 6]

    short = RingBuffer(duration_s=60)
    _append(short, 0, duration=1)
    _append(short, 1, duration=1)
    with pytest.raises(BufferError) as error:
        short.select_recent(8, now=START + timedelta(seconds=1))
    assert error.value.code == "buffer_too_short"


def test_invalid_duration_is_rejected() -> None:
    buffer = RingBuffer(duration_s=60)
    _append(buffer, 0)
    for value in (0, -1, float("nan")):
        with pytest.raises(BufferError) as error:
            buffer.select_recent(value, now=START)
        assert error.value.code == "invalid_duration"
    with pytest.raises(BufferError) as error:
        buffer.select_recent(61, now=START)
    assert error.value.code == "invalid_duration"


def test_save_last_selects_five_and_ten_seconds() -> None:
    buffer = RingBuffer(duration_s=60)
    for offset in range(12):
        _append(buffer, offset, duration=1)
    now = START + timedelta(seconds=11)
    five = buffer.select_recent(5, now=now)
    ten = buffer.select_recent(10, now=now)
    assert five[0].captured_at >= now - timedelta(seconds=5.5)
    assert ten[0].captured_at >= now - timedelta(seconds=10.5)
    assert five[-1].sequence_number == ten[-1].sequence_number == 12
    assert len(ten) > len(five)


def test_sequence_keeps_increasing_after_eviction() -> None:
    buffer = RingBuffer(duration_s=0.5)
    _append(buffer, 0)
    _append(buffer, 1)
    _append(buffer, 2)
    assert [segment.sequence_number for segment in buffer.segments()] == [3]
    added = _append(buffer, 2.2)
    assert added.sequence_number == 4
    assert added.local_id == "local:4"
