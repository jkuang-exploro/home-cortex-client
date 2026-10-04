"""Still and clip packaging, manifest identity, and local retention."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from home_cortex_client.buffer import RingBuffer
from home_cortex_client.evidence import (
    CLIP_CONTENT_TYPE,
    CLIP_MAGIC,
    EvidenceFailure,
    EvidencePackager,
    LocalEvidenceStore,
    build_manifest,
    decode_clip,
    evidence_id_for,
)
from home_cortex_client.sources import TINY_JPEG


START = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
VECTOR = {
    "embodiment_id": "embodiment:macbook-0",
    "camera_id": "camera:built_in",
    "media_type": "image",
    "captured_start": "2026-10-03T12:00:00.000+00:00",
    "captured_end": "2026-10-03T12:00:00.000+00:00",
    "duration_ms": 0,
    "sequence_start": 1042,
    "sequence_end": 1042,
    "width": 1280,
    "height": 720,
    "content_type": "image/jpeg",
    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "reason": "manual_observe",
}


def _packager(tmp_path, *, freshness_s: float = 2.0, max_items: int = 8, refresh=None):
    buffer = RingBuffer(duration_s=60)
    store = LocalEvidenceStore(tmp_path, max_items=max_items, max_age_s=3600)
    packager = EvidencePackager(
        buffer, store, embodiment_id="embodiment:macbook-0",
        freshness_s=freshness_s, clock=lambda: START, refresh=refresh,
    )
    return buffer, packager


def _fill(buffer: RingBuffer, count: int, *, step: float = 1.0, duration: float = 1.0) -> None:
    for index in range(count):
        buffer.append_frame(
            camera_id="camera:built_in",
            captured_at=START + timedelta(seconds=index * step),
            width=8,
            height=4,
            payload=TINY_JPEG,
            duration_s=duration,
        )


def test_evidence_id_is_stable_and_independent_of_a_filename() -> None:
    assert evidence_id_for(VECTOR) == "evidence:e7edfc6018adb86b739fd49624b50c9c"
    renamed = dict(VECTOR)
    renamed["sha256"] = "b" * 64
    assert evidence_id_for(renamed) != evidence_id_for(VECTOR)
    manifest = build_manifest(**VECTOR)
    assert manifest["evidence_id"] == "evidence:e7edfc6018adb86b739fd49624b50c9c"
    assert "filename" not in manifest


def test_latest_still_packages_the_newest_fresh_frame(tmp_path) -> None:
    buffer, packager = _packager(tmp_path)
    _fill(buffer, 3, step=0.2, duration=0.2)
    packaged = packager.get_latest_still(now=START + timedelta(seconds=0.4))
    manifest = packaged.manifest
    assert manifest["media_type"] == "image"
    assert manifest["duration_ms"] == 0
    assert manifest["sequence_start"] == manifest["sequence_end"] == 3
    assert manifest["captured_start"] == manifest["captured_end"]
    assert manifest["content_type"] == "image/jpeg"
    assert manifest["embodiment_id"] == "embodiment:macbook-0"
    assert manifest["camera_id"] == "camera:built_in"
    assert manifest["width"] == 8 and manifest["height"] == 4
    assert manifest["reason"] == "manual_observe"
    assert packaged.payload == TINY_JPEG
    inspected = packager.store.inspect(manifest["evidence_id"])
    assert inspected["hash_ok"] is True
    assert inspected["manifest"]["sha256"] == manifest["sha256"]


def test_stale_frame_is_not_returned_as_current(tmp_path) -> None:
    buffer, packager = _packager(tmp_path, freshness_s=1)
    _fill(buffer, 1)
    with pytest.raises(EvidenceFailure) as error:
        packager.get_latest_still(now=START + timedelta(seconds=5))
    assert error.value.code == "evidence_stale"
    assert list(tmp_path.iterdir()) == []


def test_refresh_can_replace_a_stale_frame(tmp_path) -> None:
    buffer, packager = _packager(tmp_path, freshness_s=1)

    def refresh() -> None:
        buffer.append_frame(
            camera_id="camera:built_in",
            captured_at=START + timedelta(seconds=5),
            width=8,
            height=4,
            payload=TINY_JPEG,
            duration_s=0.2,
        )

    packager.refresh = refresh
    packager.clock = lambda: START + timedelta(seconds=5)
    _fill(buffer, 1)
    packaged = packager.get_latest_still()
    assert packaged.manifest["sequence_start"] == 2
    assert packaged.manifest["captured_start"].startswith("2026-10-03T12:00:05")


def test_recent_clip_uses_actual_bounds_and_contiguous_sequences(tmp_path) -> None:
    buffer, packager = _packager(tmp_path)
    _fill(buffer, 6, step=1, duration=1)
    packaged = packager.get_recent_clip(3, now=START + timedelta(seconds=5))
    manifest = packaged.manifest
    assert manifest["media_type"] == "video_clip"
    assert manifest["content_type"] == CLIP_CONTENT_TYPE
    assert manifest["reason"] == "manual_observe_clip"
    assert manifest["sequence_start"] == 3
    assert manifest["sequence_end"] == 6
    assert manifest["sequence_start"] <= manifest["sequence_end"]
    assert manifest["duration_ms"] == 4000
    assert manifest["captured_start"].startswith("2026-10-03T12:00:02")
    assert manifest["captured_end"].startswith("2026-10-03T12:00:06")
    assert packaged.payload.startswith(CLIP_MAGIC)
    frames = decode_clip(packaged.payload)
    assert [sequence for sequence, _millis, _frame in frames] == [3, 4, 5, 6]
    inspected = packager.store.inspect(str(manifest["evidence_id"]))
    assert inspected["hash_ok"] is True


def test_clip_rejects_a_short_buffer_and_an_invalid_duration(tmp_path) -> None:
    buffer, packager = _packager(tmp_path)
    _fill(buffer, 2, step=1, duration=1)
    with pytest.raises(EvidenceFailure) as short:
        packager.get_recent_clip(8, now=START + timedelta(seconds=1))
    assert short.value.code == "buffer_too_short"
    with pytest.raises(EvidenceFailure) as invalid:
        packager.get_recent_clip(0, now=START + timedelta(seconds=1))
    assert invalid.value.code == "invalid_duration"
    with pytest.raises(EvidenceFailure) as missing:
        packager.get_recent_clip("eight", now=START)  # type: ignore[arg-type]
    assert missing.value.code == "invalid_duration"


def test_evidence_cleanup_bounds_the_directory_and_lookup_uses_the_manifest(tmp_path) -> None:
    buffer, packager = _packager(tmp_path, max_items=2)
    _fill(buffer, 1)
    first = packager.get_latest_still(now=START)
    buffer.append_frame(
        camera_id="camera:built_in",
        captured_at=START + timedelta(seconds=0.2),
        width=8, height=4, payload=TINY_JPEG, duration_s=0.2,
    )
    second = packager.get_latest_still(now=START + timedelta(seconds=0.2))
    buffer.append_frame(
        camera_id="camera:built_in",
        captured_at=START + timedelta(seconds=0.4),
        width=8, height=4, payload=TINY_JPEG, duration_s=0.2,
    )
    third = packager.get_latest_still(now=START + timedelta(seconds=0.4))
    directories = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(directories) == 2
    with pytest.raises(EvidenceFailure) as missing:
        packager.store.inspect(str(first.manifest["evidence_id"]))
    assert missing.value.code == "not_found"
    kept = packager.store.inspect(str(third.manifest["evidence_id"]))
    directory = tmp_path / str(third.manifest["evidence_id"]).split(":", 1)[1]
    moved = tmp_path / "renamed-folder"
    directory.rename(moved)
    found = packager.store.inspect(str(third.manifest["evidence_id"]))
    assert found["manifest"]["evidence_id"] == third.manifest["evidence_id"]
    assert found["hash_ok"] is True
    assert json.loads((moved / "manifest.json").read_text())["evidence_id"] == third.manifest["evidence_id"]
    assert kept["hash_ok"] is True
    assert second.manifest["evidence_id"] != third.manifest["evidence_id"]
