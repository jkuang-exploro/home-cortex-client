"""Camera source and MJPEG runtime tests. No hardware required."""
import json
import os
import time
from datetime import datetime
from urllib.request import urlopen

import pytest

from home_cortex_client.__main__ import build_parser
from home_cortex_client.config import ClientConfig
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
        while runtime.running and time.time() < deadline:
            runtime.wait(0.01)
        health = json.loads(runtime.health_json())
        assert health["running"] is False
        assert health["source_status"] == "camera_disconnected"
        assert health["failure_code"] == "camera_read_failed"
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
