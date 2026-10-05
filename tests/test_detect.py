"""Synthetic grayscale sequences for the frame-difference detector."""
from datetime import datetime, timedelta, timezone

import pytest

from home_cortex_client.config import ChangeDetectionConfig, ClientConfig
from home_cortex_client.detect import ChangeDetector, change_score, fit_grid


START = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
WIDTH = 32
HEIGHT = 24
PIXELS = WIDTH * HEIGHT


def _still(level: int = 0) -> bytes:
    return bytes([level]) * PIXELS


def _bar(offset: int) -> bytes:
    frame = bytearray(PIXELS)
    for y in range(HEIGHT):
        row = y * WIDTH
        for x in range(8):
            frame[row + offset + x] = 255
    return bytes(frame)


def _bounce(step: int) -> bytes:
    forward = list(range(0, 18, 2))
    trip = forward + forward[-2:0:-1]
    return _bar(trip[step % len(trip)])


def _push(detector: ChangeDetector, samples: list[bytes], step: float = 0.25):
    found = []
    for index, sample in enumerate(samples):
        when = START + timedelta(seconds=index * step)
        found.extend(detector.observe(when, sample, width=WIDTH, height=HEIGHT))
    return found


def _detector(**overrides) -> ChangeDetector:
    return ChangeDetector(_config(**overrides))


def _config(**overrides) -> ChangeDetectionConfig:
    values = dict(
        sample_hz=4,
        settling_s=0.2,
        minimum_event_duration_s=0.4,
        maximum_candidate_duration_s=30,
    )
    values.update(overrides)
    return ChangeDetectionConfig(**values)


def test_change_score_is_the_mean_absolute_difference() -> None:
    dark = bytes(4)
    assert change_score(dark, dark) == 0.0
    assert change_score(dark, bytes([255, 255, 0, 0])) == 0.5
    assert fit_grid(bytes([0, 10, 20, 30]), 2, 2, 1, 1) == bytes([15])


def test_static_frames_and_small_noise_emit_nothing() -> None:
    static = _push(_detector(), [_still(128)] * 12)
    assert static == []
    noise = [_still(128 if index % 2 == 0 else 130) for index in range(16)]
    assert _push(_detector(), noise) == []


def test_clear_movement_opens_one_interval_and_closes_after_settling() -> None:
    samples = [_still(), *(_bounce(index) for index in range(8)), _still(), _still(), _still()]
    detector = _detector()
    events = _push(detector, samples)
    motions = [event for event in events if event.as_mapping()["type"] == "motion"]
    assert len(motions) == 1
    motion = motions[0]
    assert motion.detector == "frame_difference"
    assert motion.started_at == START + timedelta(seconds=0.25)
    assert motion.ended_at == START + timedelta(seconds=2.25)
    assert 0.08 <= motion.peak_score <= 1
    assert motion.average_score <= motion.peak_score
    assert detector.state == "idle"
    assert detector.motion_events == 1
    body = motion.as_mapping()
    assert body["type"] == "motion"
    assert body["detector"] == "frame_difference"
    assert body["peak_score"] == motion.peak_score


def test_sampling_skips_frames_between_detector_ticks() -> None:
    detector = _detector(sample_hz=4)
    found = []
    for index in range(21):
        when = START + timedelta(milliseconds=50 * index)
        on_tick = (50 * index) % 250 == 0
        sample = _still() if on_tick else bytes([255]) * PIXELS
        found.extend(detector.observe(when, sample, width=WIDTH, height=HEIGHT))
    assert found == []
    assert detector.scored_samples == 4


def test_brief_spike_follows_the_minimum_duration() -> None:
    spike = _bar(0)
    samples = [_still(), spike, spike, spike, spike]
    kept = _push(_detector(minimum_event_duration_s=0, settling_s=0.2), samples)
    assert len(kept) == 1
    assert kept[0].started_at == kept[0].ended_at
    dropped = _detector(minimum_event_duration_s=0.4, settling_s=0.2)
    assert _push(dropped, samples) == []
    assert dropped.suppressed_short == 1


def test_full_frame_jump_is_a_scene_change() -> None:
    samples = [_still(), bytes([255]) * PIXELS, _still(), _still(), _still()]
    detector = _detector()
    events = _push(detector, samples)
    scenes = [event for event in events if event.as_mapping()["type"] == "scene_change"]
    motions = [event for event in events if event.as_mapping()["type"] == "motion"]
    assert len(scenes) >= 1
    assert scenes[0].score >= 0.45
    assert scenes[0].as_mapping()["type"] == "scene_change"
    assert motions == []
    assert detector.suppressed_short == 1


def test_long_motion_is_split_on_the_maximum_duration() -> None:
    samples = [_still(), *(_bounce(index) for index in range(360)), _still(), _still(), _still()]
    events = _push(_detector(), samples)
    motions = [event for event in events if event.as_mapping()["type"] == "motion"]
    assert len(motions) == 3
    for motion in motions:
        duration = (motion.ended_at - motion.started_at).total_seconds()
        assert motion.forced_split is True
        assert duration == pytest.approx(30.0)
    assert motions[0].ended_at == motions[1].started_at
    assert motions[1].ended_at == motions[2].started_at


def test_thresholds_stay_ordered_and_env_can_replace_them(monkeypatch) -> None:
    with pytest.raises(ValueError):
        ChangeDetectionConfig(motion_continue_threshold=0.5, motion_start_threshold=0.2)
    monkeypatch.setenv("HOME_CORTEX_CLIENT_DETECTOR_HZ", "6")
    monkeypatch.setenv("HOME_CORTEX_CLIENT_MERGE_GAP_SECONDS", "1.5")
    monkeypatch.setenv("HOME_CORTEX_CLIENT_SCENE_CHANGE_CANDIDATES", "false")
    monkeypatch.setenv("HOME_CORTEX_CLIENT_COOLDOWN_SECONDS", "0")
    loaded = ClientConfig.from_env().detection
    assert loaded.sample_hz == 6
    assert loaded.merge_gap_s == 1.5
    assert loaded.scene_change_candidates is False
    assert loaded.cooldown_s == 0
