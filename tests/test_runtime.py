"""Camera source and MJPEG runtime tests. No hardware required."""
import json
import os
import time
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen

import pytest

from home_cortex_client.__main__ import build_parser
from home_cortex_client.config import ClientConfig
from home_cortex_client.capture import CaptureError
from home_cortex_client.frames import CameraFrame, capture_timestamp
from home_cortex_client.runtime import EdgeRuntime
from home_cortex_client.sources import MacCameraSource, SyntheticCameraSource
from home_cortex_client.stream import StreamConfig


def test_configuration_reads_device_environment(monkeypatch) -> None:
    monkeypatch.setenv("HOME_CORTEX_CLIENT_DEVICE_ID", "device:test")
    monkeypatch.setenv("HOME_CORTEX_CLIENT_CAMERA_ID", "camera:test")
    monkeypatch.setenv("HOME_CORTEX_CLIENT_STREAM_PORT", "9000")
    monkeypatch.setenv("HOME_CORTEX_CLIENT_WIDTH", "320")
    config = ClientConfig.from_env()
    assert config.device_id == "device:test"
    assert config.camera_id == "camera:test"
    assert config.embodiment_id == "embodiment:macbook-0"
    assert config.stream_port == 9000
    assert config.width == 320


def test_synthetic_source_timestamps_and_dimensions() -> None:
    source = SyntheticCameraSource(width=320, height=240, fps=5)
    source.open()
    try:
        first = source.read()
        second = source.read()
    finally:
        source.close()
    assert first.width == 320 and first.height == 240
    assert first.jpeg.startswith(b"\xff\xd8")
    parsed = datetime.fromisoformat(first.captured_at)
    assert parsed.tzinfo is not None
    assert datetime.fromisoformat(second.captured_at) > parsed
    datetime.fromisoformat(capture_timestamp())


def test_sources_share_replaceable_interface() -> None:
    for source in (MacCameraSource(), SyntheticCameraSource()):
        assert source.device_id == "device:dev_macbook"
        assert source.camera_id == "camera:built_in"
        assert callable(source.open)
        assert callable(source.read)
        assert callable(source.close)


def test_stream_config_endpoint() -> None:
    config = StreamConfig(host="127.0.0.1", port=8088)
    assert config.transport == "mjpeg-http"
    assert config.endpoint == "http://127.0.0.1:8088/live.mjpg"
    assert config.viewer == "http://127.0.0.1:8088/"


def test_runtime_start_health_and_clean_shutdown() -> None:
    source = SyntheticCameraSource(fps=20)
    runtime = EdgeRuntime(source, config=StreamConfig(host="127.0.0.1", port=0), fps=20)
    endpoint = runtime.start()
    try:
        deadline = time.time() + 2
        while runtime.latest_frame() is None and time.time() < deadline:
            runtime.wait(0.02)
        assert isinstance(runtime.latest_frame(), CameraFrame)
        with urlopen(endpoint.rsplit("/", 1)[0] + "/health", timeout=2) as response:
            health = json.loads(response.read().decode())
        assert health["device_id"] == "device:dev_macbook"
        assert health["camera_id"] == "camera:built_in"
        assert health["transport"] == "mjpeg-http"
        assert health["running"] is True
        with urlopen(endpoint, timeout=2) as stream:
            chunk = stream.read(160)
        assert b"--edgeframe" in chunk or b"\xff\xd8" in chunk
    finally:
        runtime.stop()
    assert runtime.running is False


def test_capture_failure_has_stable_health_state(monkeypatch) -> None:
    class FailingSource(SyntheticCameraSource):
        def read(self) -> CameraFrame:
            raise RuntimeError("native camera detail must not cross the boundary")

    runtime = EdgeRuntime(
        FailingSource(), config=StreamConfig(host="127.0.0.1", port=0)
    )
    monkeypatch.setattr(runtime._stream, "start", lambda: None)
    runtime.start()
    try:
        deadline = time.time() + 2
        health = {}
        while time.time() < deadline:
            health = json.loads(runtime.health_json())
            if health["failure_code"]:
                break
            runtime.wait(0.01)
        assert health["running"] is True
        assert health["capture_active"] is False
        assert health["available"] is False
        assert health["source_status"] == "camera_disconnected"
        assert health["failure_code"] == "capture_interrupted"
        assert "native camera detail" not in runtime.health_json()
    finally:
        runtime.stop()


def test_stream_start_failure_closes_capture_source(monkeypatch) -> None:
    class RecordingSource(SyntheticCameraSource):
        closed = False

        def close(self) -> None:
            self.closed = True
            super().close()

    source = RecordingSource()
    runtime = EdgeRuntime(source)
    monkeypatch.setattr(
        runtime._stream, "start", lambda: (_ for _ in ()).throw(OSError("bind failed"))
    )
    with pytest.raises(OSError, match="bind failed"):
        runtime.start()
    assert source.closed is True
    assert runtime.running is False


def test_cli_defaults_are_device_local() -> None:
    parser = build_parser(ClientConfig())
    args = parser.parse_args(["--source", "synthetic", "--port", "0"])
    assert args.source == "synthetic"
    assert args.host == "127.0.0.1"
    assert args.port == 0
    assert args.embodiment_id == "embodiment:macbook-0"
    assert args.buffer_seconds == 60


def test_capture_failure_recovers_without_stopping_the_client() -> None:
    class Flaky(SyntheticCameraSource):
        def __init__(self) -> None:
            super().__init__(fps=20)
            self.remaining = 2

        def read(self) -> CameraFrame:
            if self.remaining:
                self.remaining -= 1
                raise CaptureError("capture_interrupted")
            return super().read()

    runtime = EdgeRuntime(
        Flaky(), config=StreamConfig(host="127.0.0.1", port=0), fps=20, retry_interval=0.01,
    )
    runtime.start()
    try:
        deadline = time.time() + 2
        while runtime.latest_frame() is None and time.time() < deadline:
            runtime.wait(0.01)
        assert runtime.latest_frame() is not None
        assert runtime.camera_status()["failure_code"] is None
        assert runtime.running is True
    finally:
        runtime.stop()


def test_permission_denial_keeps_the_runtime_up() -> None:
    class Denied(SyntheticCameraSource):
        def open(self) -> None:
            raise CaptureError("camera_permission_denied")

    runtime = EdgeRuntime(
        Denied(), config=StreamConfig(host="127.0.0.1", port=0), retry_interval=0.05,
    )
    runtime.start()
    try:
        deadline = time.time() + 2
        status = {}
        while time.time() < deadline:
            status = runtime.camera_status()
            if status["failure_code"]:
                break
            runtime.wait(0.01)
        assert runtime.running is True
        assert status["supported"] is True
        assert status["available"] is False
        assert status["capture_active"] is False
        assert status["failure_code"] == "camera_permission_denied"
    finally:
        runtime.stop()


def test_encoding_failure_retries_without_reopening(monkeypatch) -> None:
    class Encoder(SyntheticCameraSource):
        def __init__(self) -> None:
            super().__init__(fps=20)
            self.failed = False
            self.closes = 0

        def read(self) -> CameraFrame:
            frame = super().read()
            if not self.failed:
                self.failed = True
                raise CaptureError("encoding_failed")
            return frame

        def close(self) -> None:
            self.closes += 1
            super().close()

    source = Encoder()
    runtime = EdgeRuntime(
        source, config=StreamConfig(host="127.0.0.1", port=0), fps=20, retry_interval=0.01,
    )
    monkeypatch.setattr(runtime._stream, "start", lambda: None)
    runtime.start()
    try:
        deadline = time.time() + 2
        while runtime.latest_frame() is None and time.time() < deadline:
            runtime.wait(0.01)
        assert source.closes == 0
        assert runtime.buffer.latest() is not None
    finally:
        runtime.stop()
    assert source.closes == 1


def test_debug_api_shows_a_moving_bounded_buffer(tmp_path) -> None:
    start = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
    holder: dict = {}

    def clock() -> datetime:
        latest = holder["runtime"].buffer.latest()
        return latest.captured_at if latest is not None else start

    runtime = EdgeRuntime(
        SyntheticCameraSource(fps=20, start=start),
        config=StreamConfig(host="127.0.0.1", port=0),
        fps=20,
        buffer_seconds=2,
        freshness_seconds=2,
        evidence_dir=tmp_path,
        clock=clock,
        retry_interval=0.01,
    )
    holder["runtime"] = runtime
    runtime.start()
    try:
        base = f"http://127.0.0.1:{runtime._stream.bound_port}"
        deadline = time.time() + 3
        first = None
        while time.time() < deadline:
            status = json.loads(urlopen(base + "/debug/camera/status", timeout=2).read())
            if status["capture_active"] and status["buffer_duration"] >= 1.5:
                first = status
                break
            runtime.wait(0.02)
        assert first is not None
        assert first["supported"] is True
        assert first["available"] is True
        assert first["camera_id"] == "camera:built_in"
        assert first["latest_capture_time"]
        assert first["buffer_start_time"] < first["buffer_end_time"]
        with urlopen(base + "/debug/camera/latest-frame", timeout=2) as response:
            first_frame = response.read()
            first_sequence = int(response.headers["X-Sequence-Number"])
            assert response.headers["X-Captured-At"]
            assert response.headers["X-Fresh"] == "true"
        time.sleep(0.2)
        with urlopen(base + "/debug/camera/latest-frame", timeout=2) as response:
            second_frame = response.read()
            second_sequence = int(response.headers["X-Sequence-Number"])
        assert second_sequence > first_sequence
        assert second_frame.startswith(b"\xff\xd8")
        assert first_frame.startswith(b"\xff\xd8")
        later = json.loads(urlopen(base + "/debug/camera/status", timeout=2).read())
        assert later["latest_capture_time"] >= first["latest_capture_time"]
        assert later["buffer_duration"] <= 2.2
        listed = json.loads(urlopen(base + "/debug/camera/buffer", timeout=2).read())
        assert listed["segments"]
        assert listed["segments"][0]["sequence_number"] <= listed["segments"][-1]["sequence_number"]
        for seconds in (1,):
            request = Request(f"{base}/debug/camera/save-last?seconds={seconds}", data=b"", method="POST")
            with urlopen(request, timeout=2) as response:
                saved = response.read()
                assert response.status == 200
                assert saved.startswith(b"HCCLIP1")
                assert response.headers["X-Captured-Start"]
                assert response.headers["X-Captured-End"]
                assert int(response.headers["X-Sequence-Start"]) <= int(response.headers["X-Sequence-End"])
        evidence = json.loads(urlopen(base + "/debug/evidence/latest", timeout=2).read())
        assert evidence["hash_ok"] is True
        assert evidence["manifest"]["media_type"] == "image"
        evidence_id = evidence["manifest"]["evidence_id"]
        inspected = json.loads(urlopen(base + "/debug/evidence/" + evidence_id, timeout=2).read())
        assert inspected["hash_ok"] is True
        assert inspected["manifest"]["sha256"] == evidence["manifest"]["sha256"]
    finally:
        runtime.stop()


@pytest.mark.manual
@pytest.mark.skipif(
    os.environ.get("HOME_CORTEX_CLIENT_CAMERA_SMOKE") != "1",
    reason="opt-in physical camera smoke",
)
def test_mac_camera_smoke() -> None:
    source = MacCameraSource()
    source.open()
    try:
        frame = source.read()
    finally:
        source.close()
    assert frame.width > 0 and frame.height > 0
    assert frame.jpeg.startswith(b"\xff\xd8")
    datetime.fromisoformat(frame.captured_at)
