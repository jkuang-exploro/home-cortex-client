"""Glue for a camera source and encoded MJPEG preview."""
from __future__ import annotations

import json
import threading
from typing import Any

from .frames import CameraFrame, CameraSource
from .stream import MJPEGStreamServer, StreamConfig


class EdgeRuntime:
    def __init__(
        self,
        source: CameraSource,
        *,
        config: StreamConfig | None = None,
        fps: float = 10.0,
    ) -> None:
        self.source = source
        self.config = config or StreamConfig()
        self.fps = fps
        self.frame_interval = 1.0 / fps if fps > 0 else 0.1
        self._lock = threading.Lock()
        self._latest: CameraFrame | None = None
        self._stop = threading.Event()
        self._frames = 0
        self._failure_code: str | None = None
        self._capture_thread: threading.Thread | None = None
        self._stream = MJPEGStreamServer(self, self.config)

    @property
    def running(self) -> bool:
        return not self._stop.is_set() and self._capture_thread is not None

    def start(self) -> str:
        self._stop.clear()
        self._failure_code = None
        self.source.open()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        try:
            self._stream.start()
        except Exception:
            self.stop()
            raise
        return self._stream.endpoint

    def stop(self) -> None:
        self._stop.set()
        self._stream.stop()
        thread = self._capture_thread
        self._capture_thread = None
        if thread is not None:
            thread.join(timeout=2)
        self.source.close()

    def wait(self, timeout: float) -> None:
        self._stop.wait(timeout)

    def latest_frame(self) -> CameraFrame | None:
        with self._lock:
            return self._latest

    @property
    def viewer(self) -> str:
        return self._stream.viewer

    def health_json(self) -> str:
        frame = self.latest_frame()
        payload: dict[str, Any] = {
            "device_id": self.source.device_id,
            "camera_id": self.source.camera_id,
            "transport": self.config.transport,
            "stream": self._stream.endpoint,
            "viewer": self._stream.viewer,
            "frames": self._frames,
            "running": self.running,
            "source_status": (
                "camera_disconnected"
                if self._failure_code is not None
                else "online" if self.running else "offline"
            ),
            "failure_code": self._failure_code,
            "last_captured_at": None if frame is None else frame.captured_at,
            "width": None if frame is None else frame.width,
            "height": None if frame is None else frame.height,
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.source.read()
            except Exception:
                self._failure_code = "camera_read_failed"
                self._stop.set()
                return
            with self._lock:
                self._latest = frame
                self._frames += 1
            self._stop.wait(self.frame_interval)
