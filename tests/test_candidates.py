"""Local candidate clips from synthetic visual-change intervals."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from home_cortex_client.buffer import RingBuffer
from home_cortex_client.candidates import CandidateEngine, CandidateFailure, CandidateStore
from home_cortex_client.config import ChangeDetectionConfig
from home_cortex_client.debug import dispatch
from home_cortex_client.detect import ChangeDetector, MotionEvent, SceneChangeEvent
from home_cortex_client.debug_cli import main as debug_main
from home_cortex_client.evidence import decode_clip
from home_cortex_client.frames import CameraFrame
from home_cortex_client.runtime import EdgeRuntime
from home_cortex_client.sources import TINY_JPEG, SyntheticCameraSource
from home_cortex_client.stream import MJPEGStreamServer, StreamConfig


START = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
WIDTH = 32
HEIGHT = 24
PIXELS = WIDTH * HEIGHT
STEP = 0.25


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


def _config(**overrides) -> ChangeDetectionConfig:
    values = dict(
        sample_hz=4,
        motion_start_threshold=0.08,
        motion_continue_threshold=0.04,
        scene_change_threshold=0.45,
        settling_s=0.2,
        minimum_event_duration_s=0.4,
        pre_roll_s=1.0,
        post_roll_s=1.0,
        merge_gap_s=2.0,
        minimum_peak_score=0.08,
        maximum_candidate_duration_s=30.0,
        cooldown_s=0.0,
        max_candidates=32,
        max_candidate_age_s=3600.0,
        max_candidate_disk_bytes=32 * 1024 * 1024,
        scene_change_candidates=True,
    )
    values.update(overrides)
    return ChangeDetectionConfig(**values)


def _drive(samples: list[bytes], root, config: ChangeDetectionConfig | None = None):
    chosen = config or _config()
    span = max(len(samples) * STEP + 5, 30)
    buffer = RingBuffer(duration_s=span, max_bytes=64 * 1024 * 1024)
    detector = ChangeDetector(chosen)
    engine = CandidateEngine(
        buffer,
        embodiment_id="embodiment:macbook-0",
        camera_id="camera:built_in",
        root=root,
        config=chosen,
    )
    for index, sample in enumerate(samples):
        when = START + timedelta(seconds=index * STEP)
        buffer.append_frame(
            camera_id="camera:built_in",
            captured_at=when,
            width=1,
            height=1,
            payload=TINY_JPEG,
            duration_s=STEP,
        )
        for signal in detector.observe(when, sample, width=WIDTH, height=HEIGHT):
            engine.ingest(signal)
        interval = detector.active_interval()
        if interval is None:
            engine.note_motion(None, None)
        else:
            engine.note_motion(interval[0], interval[1])
        engine.advance(when)
    return detector, engine


def _burst(frames: int = 8) -> list[bytes]:
    return [_bounce(index) for index in range(frames)]


def _gap(frames: int) -> list[bytes]:
    return [_still() for _ in range(frames)]


def test_motion_candidate_includes_pre_roll_post_roll_and_stays_local(tmp_path) -> None:
    samples = [*_gap(8), *_burst(8), *_gap(12)]
    _detector, engine = _drive(samples, tmp_path)
    records = engine.store.local_records()
    assert len(records) == 1
    record = records[0]
    assert record["embodiment_id"] == "embodiment:macbook-0"
    assert record["trigger_type"] == "motion"
    assert record["detector"] == "frame_difference"
    assert record["upload_state"] == "local_only"
    assert record["status"] == "local"
    assert record["peak_score"] >= 0.08
    started = datetime.fromisoformat(str(record["trigger_started_at"]))
    ended = datetime.fromisoformat(str(record["trigger_ended_at"]))
    clip_start = datetime.fromisoformat(str(record["clip_start"]))
    clip_end = datetime.fromisoformat(str(record["clip_end"]))
    assert clip_start <= started
    assert clip_start <= started - timedelta(seconds=1.0 - STEP)
    assert clip_start >= started - timedelta(seconds=1.0 + STEP)
    assert clip_end >= ended + timedelta(seconds=1.0)
    inspected = engine.store.inspect(str(record["candidate_id"]))
    payload = engine.store.media(str(record["candidate_id"]))
    assert payload.startswith(b"HCCLIP1")
    assert inspected["hash_ok"] is True
    assert inspected["manifest_ok"] is True
    assert inspected["evidence_available"] is True
    assert inspected["manifest"]["reason"] == "visual_change"
    assert inspected["manifest"]["evidence_id"] == record["evidence_id"]
    assert inspected["manifest"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert engine.stats()["upload_states"] == ["local_only"]
    assert inspected["provenance"]["detector_config"]["sample_hz"] == 4
    assert inspected["provenance"]["transfer"]["has_left_client"] is False
    assert inspected["provenance"]["media_sha256"] == inspected["manifest"]["sha256"]


def test_event_cli_lists_shows_and_exports_local_clip(tmp_path, capsys) -> None:
    _detector, engine = _drive(
        [*_gap(8), *_burst(8), *_gap(12)], tmp_path / "candidates",
    )
    candidate_id = str(engine.store.local_records()[0]["candidate_id"])
    runtime = SimpleNamespace(candidates=engine, event_stats=lambda: {
        "frames_sampled": 28, "events_triggered": 1,
        "candidates_produced": 1, "candidate_bytes_retained": engine.store.disk_bytes(),
        "old_candidates_expired": 0, "transfer": "local_only",
    })
    server = MJPEGStreamServer(runtime, StreamConfig(port=0))
    server.start()
    try:
        prefix = ["--base-url", server.viewer.rstrip("/")]
        assert debug_main([*prefix, "events", "list"]) == 0
        rows = json.loads(capsys.readouterr().out)["events"]
        assert len(rows) == 1
        assert rows[0]["candidate_id"] == candidate_id
        assert rows[0]["status"] == "LOCAL"
        assert rows[0]["transfer_state"] == "local_only"

        assert debug_main([*prefix, "events", "show", candidate_id]) == 0
        shown = json.loads(capsys.readouterr().out)
        assert shown["hash_ok"] is True
        assert shown["provenance"]["sequence_range"]["start"] >= 1
        assert shown["provenance"]["requested_interval"]["start"]
        assert shown["provenance"]["local_clip_path"]

        assert debug_main([*prefix, "events", "stats"]) == 0
        assert json.loads(capsys.readouterr().out)["frames_sampled"] == 28

        output = tmp_path / "export"
        assert debug_main([
            *prefix, "events", "clip", candidate_id, "--output", str(output),
        ]) == 0
        exported = json.loads(capsys.readouterr().out)
        assert exported["hash_ok"] is True
        assert exported["has_left_client"] is False
        assert len(decode_clip((output / "clip.hcc").read_bytes())) == exported["frames"]
        assert (output / "index.html").is_file()
        assert (output / "frame-000000.jpg").read_bytes() == TINY_JPEG
    finally:
        server.stop()


def test_candidate_inspection_survives_client_restart(tmp_path) -> None:
    _detector, engine = _drive(
        [*_gap(8), *_burst(8), *_gap(12)], tmp_path,
        _config(max_candidate_age_s=1_000_000_000),
    )
    candidate_id = str(engine.store.local_records()[0]["candidate_id"])
    recovered = CandidateStore(
        tmp_path, max_items=32, max_age_s=1_000_000_000,
        max_bytes=32 * 1024 * 1024,
    )
    assert recovered.inspect(candidate_id)["hash_ok"] is True
    assert recovered.media(candidate_id) == engine.store.media(candidate_id)


def test_candidate_manifest_tampering_is_visible_at_inspection(tmp_path) -> None:
    _detector, engine = _drive([*_gap(8), *_burst(8), *_gap(12)], tmp_path)
    candidate_id = str(engine.store.local_records()[0]["candidate_id"])
    path = Path(str(engine.store.inspect(candidate_id)["provenance"]["local_clip_path"])).parent
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["camera_id"] = "camera:forged"
    (path / "manifest.json").write_text(json.dumps(manifest))
    inspected = engine.store.inspect(candidate_id)
    assert inspected["manifest_ok"] is False
    assert inspected["hash_ok"] is False


def test_oversize_candidate_does_not_exceed_disk_cap(tmp_path) -> None:
    _detector, engine = _drive(
        [*_gap(4), bytes([255]) * PIXELS, *_gap(12)], tmp_path,
        _config(max_candidate_disk_bytes=1),
    )
    assert engine.store.disk_bytes() == 0
    assert engine.store.local_records() == []
    assert engine.store.expired_by_bytes >= 1


def test_short_gap_merges_and_a_long_gap_stays_separate(tmp_path) -> None:
    merged_samples = [*_gap(2), *_burst(8), *_gap(2), *_burst(8), *_gap(12)]
    merged_detector, merged = _drive(merged_samples, tmp_path / "merged")
    assert merged_detector.motion_events == 2
    assert len(merged.store.local_records()) == 1
    assert merged.merged_extensions == 1

    separate_samples = [*_gap(2), *_burst(8), *_gap(16), *_burst(8), *_gap(12)]
    separate_detector, separate = _drive(separate_samples, tmp_path / "separate")
    assert separate_detector.motion_events == 2
    assert len(separate.store.local_records()) == 2
    assert separate.merged_extensions == 0


def test_tiny_noise_and_disabled_scene_change_leave_no_candidate(tmp_path) -> None:
    noise = [_still(128 if index % 2 == 0 else 130) for index in range(20)]
    _detector, engine = _drive(noise, tmp_path / "noise")
    assert engine.store.local_records() == []
    flash = [*_gap(2), bytes([255]) * PIXELS, *_gap(8)]
    _scene_detector, disabled = _drive(
        flash, tmp_path / "disabled", _config(scene_change_candidates=False),
    )
    assert _scene_detector.scene_changes >= 1
    assert disabled.store.local_records() == []


def test_scene_change_inside_motion_stays_on_that_candidate(tmp_path) -> None:
    samples = [*_gap(2), *_burst(4), bytes([255]) * PIXELS, *_burst(4), *_gap(12)]
    detector, engine = _drive(samples, tmp_path)
    records = engine.store.local_records()
    assert detector.scene_changes >= 1
    assert len(records) == 1
    assert records[0]["trigger_type"] == "motion"
    assert records[0]["scene_change_score"] is not None
    assert records[0]["scene_change_score"] >= 0.45
    assert engine.stats()["scene_candidates"] == 0
    assert engine.folded_scenes >= 1


def test_isolated_scene_change_can_become_a_candidate(tmp_path) -> None:
    samples = [*_gap(4), bytes([255]) * PIXELS, *_gap(12)]
    detector, engine = _drive(samples, tmp_path)
    records = engine.store.local_records()
    assert detector.motion_events == 0
    assert len(records) == 1
    assert records[0]["trigger_type"] == "scene_change"
    assert records[0]["upload_state"] == "local_only"
    assert records[0]["peak_score"] >= 0.45


def test_long_motion_becomes_bounded_candidates(tmp_path) -> None:
    samples = [_still(), *(_bounce(index) for index in range(360)), *_gap(12)]
    detector, engine = _drive(samples, tmp_path, _config(pre_roll_s=1, post_roll_s=1))
    records = engine.store.local_records()
    assert detector.motion_events == 3
    assert len(records) == 3
    durations = []
    for record in records:
        assert record["trigger_type"] == "motion"
        started = datetime.fromisoformat(str(record["trigger_started_at"]))
        ended = datetime.fromisoformat(str(record["trigger_ended_at"]))
        durations.append((ended - started).total_seconds())
    assert durations == pytest.approx([30.0, 30.0, 30.0])
    assert engine.merged_extensions == 0


def test_duplicate_overlapping_scene_is_suppressed(tmp_path) -> None:
    config = _config(pre_roll_s=0, post_roll_s=0, minimum_peak_score=0)
    buffer = RingBuffer(duration_s=30)
    buffer.append_frame(
        camera_id="camera:built_in",
        captured_at=START,
        width=1,
        height=1,
        payload=TINY_JPEG,
        duration_s=1,
    )
    engine = CandidateEngine(
        buffer,
        embodiment_id="embodiment:macbook-0",
        camera_id="camera:built_in",
        root=tmp_path,
        config=config,
    )
    scene = SceneChangeEvent(START, 0.9)
    engine.ingest(scene)
    engine.ingest(scene)
    engine.advance(START)
    assert len(engine.store.local_records()) == 1
    assert engine.suppressed_duplicates == 1


def test_cooldown_suppresses_only_the_following_start(tmp_path) -> None:
    config = _config(
        pre_roll_s=0, post_roll_s=0, merge_gap_s=0, cooldown_s=2,
        minimum_event_duration_s=0, minimum_peak_score=0,
    )
    buffer = RingBuffer(duration_s=30)
    for offset in (0, 4):
        buffer.append_frame(
            camera_id="camera:built_in",
            captured_at=START + timedelta(seconds=offset),
            width=1,
            height=1,
            payload=TINY_JPEG,
            duration_s=1,
        )
    engine = CandidateEngine(
        buffer,
        embodiment_id="embodiment:macbook-0",
        camera_id="camera:built_in",
        root=tmp_path,
        config=config,
    )

    def motion(start: float, end: float) -> MotionEvent:
        return MotionEvent(
            START + timedelta(seconds=start),
            START + timedelta(seconds=end),
            0.4,
            0.3,
            4,
        )

    engine.ingest(motion(0, 1))
    engine.advance(START + timedelta(seconds=1))
    engine.ingest(motion(1.2, 1.8))
    engine.advance(START + timedelta(seconds=1.8))
    engine.ingest(motion(4, 5))
    engine.advance(START + timedelta(seconds=5))
    assert engine.suppressed_cooldown == 1
    assert len(engine.store.local_records()) == 2


def test_retention_expires_metadata_and_bytes_together(tmp_path) -> None:
    config = _config(
        pre_roll_s=0, post_roll_s=0, merge_gap_s=0, minimum_peak_score=0,
        max_candidates=2, max_candidate_age_s=100_000, max_candidate_disk_bytes=10_000_000,
    )
    buffer = RingBuffer(duration_s=60)
    for offset in (0, 10, 20):
        buffer.append_frame(
            camera_id="camera:built_in",
            captured_at=START + timedelta(seconds=offset),
            width=1,
            height=1,
            payload=TINY_JPEG,
            duration_s=1,
        )
    engine = CandidateEngine(
        buffer,
        embodiment_id="embodiment:macbook-0",
        camera_id="camera:built_in",
        root=tmp_path,
        config=config,
    )
    saved = []
    for offset in (0, 10, 20):
        when = START + timedelta(seconds=offset)
        engine.ingest(SceneChangeEvent(when, 0.8))
        packaged = engine.advance(when)
        saved.extend(packaged)
    assert len(saved) == 3
    local = engine.store.local_records()
    assert len(local) == 2
    oldest = str(saved[0]["candidate_id"])
    newest = str(saved[-1]["candidate_id"])
    assert local[-1]["candidate_id"] == newest
    assert engine.store.get(oldest)["status"] == "expired"
    with pytest.raises(CandidateFailure) as missing:
        engine.store.media(oldest)
    assert missing.value.code == "expired"
    inspected = engine.store.inspect(oldest)
    assert inspected["evidence_available"] is False
    assert engine.store.disk_bytes() == sum(
        len(engine.store.media(str(record["candidate_id"]))) for record in local
    )


def test_age_expiry_removes_the_tombstone_on_a_later_save(tmp_path) -> None:
    config = _config(
        pre_roll_s=0, post_roll_s=0, merge_gap_s=0, minimum_peak_score=0,
        max_candidates=8, max_candidate_age_s=5, max_candidate_disk_bytes=10_000_000,
    )
    buffer = RingBuffer(duration_s=60)
    for offset in (0, 10, 16):
        buffer.append_frame(
            camera_id="camera:built_in",
            captured_at=START + timedelta(seconds=offset),
            width=1,
            height=1,
            payload=TINY_JPEG,
            duration_s=1,
        )
    engine = CandidateEngine(
        buffer,
        embodiment_id="embodiment:macbook-0",
        camera_id="camera:built_in",
        root=tmp_path,
        config=config,
    )
    first_when = START
    engine.ingest(SceneChangeEvent(first_when, 0.8))
    first = engine.advance(first_when)[0]
    second_when = START + timedelta(seconds=10)
    engine.ingest(SceneChangeEvent(second_when, 0.8))
    engine.advance(second_when)
    assert engine.store.get(str(first["candidate_id"]))["status"] == "expired"
    third_when = START + timedelta(seconds=16)
    engine.ingest(SceneChangeEvent(third_when, 0.8))
    engine.advance(third_when)
    with pytest.raises(CandidateFailure) as missing:
        engine.store.get(str(first["candidate_id"]))
    assert missing.value.code == "not_found"


def test_empty_buffer_counts_a_miss_without_raising(tmp_path) -> None:
    engine = CandidateEngine(
        RingBuffer(duration_s=30),
        embodiment_id="embodiment:macbook-0",
        camera_id="camera:built_in",
        root=tmp_path,
        config=_config(pre_roll_s=0, post_roll_s=0, merge_gap_s=0, minimum_event_duration_s=0),
    )
    engine.ingest(MotionEvent(START, START + timedelta(seconds=1), 0.5, 0.4, 4))
    assert engine.advance(START + timedelta(seconds=1)) == []
    assert engine.missed_buffer == 1


def test_runtime_scores_the_capture_path_and_keeps_detector_errors_local(tmp_path) -> None:
    runtime = EdgeRuntime(
        SyntheticCameraSource(),
        fps=4,
        buffer_seconds=30,
        evidence_dir=tmp_path / "evidence",
        candidate_dir=tmp_path / "candidates",
        detection=_config(pre_roll_s=0.5, post_roll_s=0.5, merge_gap_s=0.5),
        embodiment_id="embodiment:macbook-0",
    )
    runtime._accept(CameraFrame(
        captured_at=START.isoformat(timespec="milliseconds"),
        width=1,
        height=1,
        jpeg=TINY_JPEG,
        sample=b"bad",
        sample_width=WIDTH,
        sample_height=HEIGHT,
    ))
    assert runtime.camera_status()["failure_code"] is None
    assert runtime._frames == 1
    assert runtime.detector_status()["error"]
    samples = [_still(), *_burst(8), *_gap(8)]
    for index, sample in enumerate(samples, start=1):
        when = START + timedelta(seconds=index * STEP)
        runtime._accept(CameraFrame(
            captured_at=when.isoformat(timespec="milliseconds"),
            width=1,
            height=1,
            jpeg=TINY_JPEG,
            sample=sample,
            sample_width=WIDTH,
            sample_height=HEIGHT,
        ))
    records = runtime.candidates.store.local_records()
    assert len(records) == 1
    assert records[0]["upload_state"] == "local_only"
    assert runtime.camera_status()["failure_code"] is None
    stats = runtime.event_stats()
    assert stats["frames_sampled"] == len(samples)
    assert stats["events_triggered"] >= 1
    assert stats["candidates_produced"] == 1
    assert stats["candidate_bytes_retained"] > 0
    assert stats["transfer"] == "local_only"
    assert runtime.detector_status()["error"] is None
    listed = json.loads(dispatch(runtime, "GET", "/debug/candidates", {})[3])
    assert listed["stats"]["candidates"] == 1
    candidate_id = records[0]["candidate_id"]
    inspected = json.loads(dispatch(runtime, "GET", "/debug/candidates/" + str(candidate_id), {})[3])
    assert inspected["hash_ok"] is True
    status = json.loads(dispatch(runtime, "GET", "/debug/detector/status", {})[3])
    assert status["detector"] == "frame_difference"
    assert status["motion_events"] == 1
    missing = dispatch(runtime, "GET", "/debug/candidates/candidate:missing", {})
    assert missing[0] == 404


def test_active_scene_reduces_fragmented_candidates(tmp_path) -> None:
    motion = [_bounce(index) for index in range(360)]
    motion[80] = bytes([255]) * PIXELS
    samples = [_still(), *motion, *_gap(12)]
    for _cycle in range(12):
        for burst in range(3):
            samples.extend(_burst(32))
            samples.extend(_gap(2 if burst < 2 else 16))
    samples.extend([bytes([255]) * PIXELS, *_gap(12)])
    samples.extend(_still(2 if index % 2 else 0) for index in range(120))
    target = int(600 / STEP) + 2
    if len(samples) < target:
        samples.extend(_gap(target - len(samples)))
    detector, engine = _drive(samples, tmp_path)
    stats = engine.stats()
    durations = []
    payload_bytes = 0
    for record in engine.store.local_records():
        assert record["upload_state"] == "local_only"
        assert record["status"] == "local"
        started = datetime.fromisoformat(str(record["trigger_started_at"]))
        ended = datetime.fromisoformat(str(record["trigger_ended_at"]))
        durations.append((ended - started).total_seconds())
        payload_bytes += len(engine.store.media(str(record["candidate_id"])))
    report = {
        "seconds": round((len(samples) - 1) * STEP, 3),
        "raw_motion_events": detector.motion_events,
        "raw_scene_changes": detector.scene_changes,
        "suppressed_short": detector.suppressed_short,
        "trigger_durations": [round(item, 3) for item in durations],
        **stats,
    }
    detail = json.dumps(report, default=str)
    assert report["seconds"] >= 600, detail
    assert report["raw_motion_events"] > report["candidates"], detail
    assert report["merged_extensions"] >= 12, detail
    assert max(durations) <= 30.0 + 1e-6, detail
    assert stats["missed_buffer"] == 0, detail
    assert stats["disk_bytes"] == payload_bytes, detail
    assert stats["upload_states"] == ["local_only"], detail
    assert report["raw_motion_events"] == 39, detail
    assert report["candidates"] == 16, detail
    assert report["scene_candidates"] == 1, detail
    assert report["folded_scenes"] == 2, detail
    assert report["suppressed_duplicates"] == 1, detail
    assert report["longest_trigger_s"] == 30.0, detail
