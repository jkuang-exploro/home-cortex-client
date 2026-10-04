"""Explicit observation commands are fulfilled from the local packager."""
from datetime import datetime, timedelta, timezone

from home_cortex_client.buffer import RingBuffer
from home_cortex_client.commands import fulfill_pending
from home_cortex_client.evidence import EvidencePackager, LocalEvidenceStore
from home_cortex_client.sources import TINY_JPEG


START = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


class FakeSession:
    def __init__(self, commands: list[dict]) -> None:
        self.commands = commands
        self.submitted = []
        self.failed = []

    def poll_commands(self) -> list[dict]:
        pending = self.commands
        self.commands = []
        return pending

    def submit_observation(self, command_id: str, manifest: dict, media: bytes) -> dict:
        self.submitted.append((command_id, manifest, media))
        return {"ok": True}

    def submit_observation_failure(self, command_id: str, code: str, message: str) -> dict:
        self.failed.append((command_id, code, message))
        return {"ok": False}


def _ready(tmp_path, *, freshness_s: float = 2.0):
    buffer = RingBuffer(duration_s=60)
    store = LocalEvidenceStore(tmp_path, max_items=8, max_age_s=3600)
    packager = EvidencePackager(
        buffer, store, embodiment_id="embodiment:macbook-0",
        freshness_s=freshness_s, clock=lambda: START, refresh=None,
    )
    return buffer, packager


def test_observe_and_clip_commands_submit_packaged_bytes(tmp_path) -> None:
    buffer, packager = _ready(tmp_path)
    for index in range(6):
        buffer.append_frame(
            camera_id="camera:built_in",
            captured_at=START + timedelta(seconds=index),
            width=8, height=4, payload=TINY_JPEG, duration_s=1,
        )
    packager.clock = lambda: START + timedelta(seconds=5)
    session = FakeSession([
        {"command_id": "observation-command:1", "operation": "vision.observe"},
        {
            "command_id": "observation-command:2",
            "operation": "vision.observe_clip",
            "duration_seconds": 3,
        },
    ])
    assert fulfill_pending(session, packager) == 2
    assert session.failed == []
    still_id, still_manifest, still_media = session.submitted[0]
    clip_id, clip_manifest, clip_media = session.submitted[1]
    assert still_id == "observation-command:1"
    assert still_manifest["media_type"] == "image"
    assert still_media.startswith(b"\xff\xd8")
    assert clip_id == "observation-command:2"
    assert clip_manifest["media_type"] == "video_clip"
    assert clip_manifest["duration_ms"] > 0
    assert clip_media.startswith(b"HCCLIP1")


def test_stale_and_empty_commands_keep_distinct_codes(tmp_path) -> None:
    buffer, packager = _ready(tmp_path, freshness_s=1)
    buffer.append_frame(
        camera_id="camera:built_in",
        captured_at=START, width=8, height=4, payload=TINY_JPEG, duration_s=1,
    )
    packager.clock = lambda: START + timedelta(seconds=10)
    stale = FakeSession([
        {"command_id": "observation-command:9", "operation": "vision.observe"},
    ])
    fulfill_pending(stale, packager)
    assert stale.failed[0][1] == "evidence_stale"

    _empty, bare = _ready(tmp_path / "empty")
    missing = FakeSession([
        {"command_id": "observation-command:4", "operation": "vision.observe"},
    ])
    fulfill_pending(missing, bare)
    assert missing.failed[0][1] == "camera_unavailable"
