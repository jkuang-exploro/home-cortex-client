"""Glue for a camera source, local ring buffer, preview, and visual-change candidates."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .buffer import BufferError, RingBuffer
from .candidates import CandidateEngine
from .capture import CaptureError
from .config import ChangeDetectionConfig
from .detect import ChangeDetector
from .evidence import EvidencePackager, LocalEvidenceStore
from .frames import CameraFrame, CameraSource
from .stream import MJPEGStreamServer, StreamConfig


class EdgeRuntime:
    def __init__(
        self,
        source: CameraSource,
        *,
        config: StreamConfig | None = None,
        fps: float = 10.0,
        buffer_seconds: float = 60.0,
        buffer_max_bytes: int = 256 * 1024 * 1024,
        freshness_seconds: float = 2.0,
        evidence_dir: str | Path | None = None,
        evidence_max_items: int = 8,
        evidence_max_age_s: float = 3600.0,
        embodiment_id: str = "embodiment:macbook-0",
        detection: ChangeDetectionConfig | None = None,
        candidate_dir: str | Path | None = None,
        clock: Callable[[], datetime] | None = None,
        retry_interval: float = 0.5,
    ) -> None:
        self.source = source
        self.config = config or StreamConfig()
        self.fps = fps
        self.frame_interval = 1.0 / fps if fps > 0 else 0.1
        self.freshness_s = freshness_seconds
        self.clock = clock or (lambda: datetime.now().astimezone())
        self.retry_interval = retry_interval
        self.buffer = RingBuffer(duration_s=buffer_seconds, max_bytes=buffer_max_bytes)
        root = Path(evidence_dir) if evidence_dir is not None else (
            Path(tempfile.gettempdir()) / "home-cortex-client" / str(os.getpid()) / "evidence"
        )
        self.evidence_store = LocalEvidenceStore(
            root, max_items=evidence_max_items, max_age_s=evidence_max_age_s,
        )
        self.packager = EvidencePackager(
            self.buffer,
            self.evidence_store,
            embodiment_id=embodiment_id,
            freshness_s=freshness_seconds,
            clock=self.clock,
            refresh=self.wait_for_newer_frame,
        )
        self.detection = detection or ChangeDetectionConfig()
        candidate_root = (
            Path(candidate_dir) if candidate_dir is not None
            else root.with_name(root.name + "-candidates")
        )
        self.detector = ChangeDetector(self.detection)
        self.candidates = CandidateEngine(
            self.buffer,
            embodiment_id=embodiment_id,
            camera_id=source.camera_id,
            root=candidate_root,
            config=self.detection,
        )
        self._lock = threading.Lock()
        self._detector_lock = threading.Lock()
        self._detector_error: str | None = None
        self._latest: CameraFrame | None = None
        self._stop = threading.Event()
        self._frames = 0
        self._failure_code: str | None = None
        self._device_open = False
        self._capture_thread: threading.Thread | None = None
        self._analyzer: Any | None = None
        self._session_capabilities = ("vision.observe",)
        self._stream = MJPEGStreamServer(self, self.config)

    @property
    def running(self) -> bool:
        return not self._stop.is_set() and self._capture_thread is not None

    def start(self) -> str:
        self._stop.clear()
        with self._lock:
            self._failure_code = None
        self._open_device()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        try:
            self._stream.start()
        except Exception:
            self.stop()
            raise
        return self._stream.endpoint

    def attach_analyzer(self, analyzer: Any) -> None:
        """Remember a background semantic worker. ``start`` does not create one."""
        self._analyzer = analyzer

    def stop(self) -> None:
        self._stop.set()
        analyzer = self._analyzer
        if analyzer is not None:
            analyzer.stop()
        self._stream.stop()
        thread = self._capture_thread
        self._capture_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        self._close_device()

    def wait(self, timeout: float) -> None:
        self._stop.wait(timeout)

    def latest_frame(self) -> CameraFrame | None:
        with self._lock:
            return self._latest

    def wait_for_newer_frame(self) -> None:
        """Wait inside the freshness window for the capture loop to append a frame."""
        import time

        before = self.buffer.latest_sequence()
        deadline = time.monotonic() + self.freshness_s
        while time.monotonic() < deadline and not self._stop.is_set():
            latest = self.buffer.latest_sequence()
            if latest and latest != before:
                return
            self._stop.wait(0.02)

    @property
    def viewer(self) -> str:
        return self._stream.viewer

    def camera_status(self) -> dict[str, Any]:
        with self._lock:
            failure = self._failure_code
            device_open = self._device_open
        latest = self.buffer.latest()
        fresh = False
        if latest is not None and failure is None and device_open:
            age = (self.clock() - latest.captured_at).total_seconds()
            fresh = -2.0 <= age <= self.freshness_s
        start, end, duration = self.buffer.span()
        active = bool(self.running and device_open and failure is None)
        return {
            "supported": True,
            "available": bool(fresh and active),
            "capture_active": active,
            "camera_id": self.source.camera_id,
            "resolution": None if latest is None else {"width": latest.width, "height": latest.height},
            "latest_capture_time": None if latest is None else latest.captured_at.isoformat(timespec="milliseconds"),
            "buffer_start_time": start,
            "buffer_end_time": end,
            "buffer_duration": duration,
            "failure_code": failure,
        }

    def detector_status(self) -> dict[str, Any]:
        with self._detector_lock:
            body = self.detector.status()
            body["error"] = self._detector_error
            return body

    def set_session_capabilities(self, names: tuple[str, ...] | list[str]) -> None:
        """Names the embodiment record is configured to accept. Hardware does not choose them."""
        allowed = ("vision.observe", "vision.observe_clip", "vision.autonomous_promotion")
        chosen = tuple(name for name in allowed if name in set(names))
        if len(chosen) != len(set(names)):
            raise ValueError("session capabilities must be canonical vision function names")
        self._session_capabilities = chosen

    def session_advertisement(self) -> list[str]:
        """Capabilities this client can perform now, limited to the configured names."""
        local = self.local_capabilities()
        return [
            name for name in self._session_capabilities if local.get(name)
        ]

    def local_capabilities(self) -> dict[str, bool]:
        """Canonical support flags. The embodiment id does not select them."""
        from .policy import canonical_capabilities

        camera = bool(self.camera_status()["available"])
        semantic = self._analyzer is not None
        return canonical_capabilities(
            observe=camera,
            observe_clip=camera,
            semantic_filter=semantic,
            autonomous_promotion=semantic,
        )

    def semantic_index(self) -> dict[str, Any]:
        analyzer = self._analyzer
        if analyzer is None:
            return {"analyzer": "off", "results": []}
        return {"analyzer": analyzer.model.model_id, "results": analyzer.store.list_results()}

    def semantic_result(self, candidate_id: str) -> dict[str, Any] | None:
        analyzer = self._analyzer
        if analyzer is None:
            return None
        try:
            return analyzer.store.load_mapping(candidate_id)
        except ValueError:
            return None

    def candidate_index(self) -> dict[str, Any]:
        with self._detector_lock:
            return {
                "candidates": self.candidates.store.records(),
                "stats": self.candidates.stats(),
            }

    def event_stats(self) -> dict[str, Any]:
        with self._detector_lock:
            detector = self.detector.status()
            candidates = self.candidates.stats()
            return {
                "camera_frames_captured": self._frames,
                "frames_sampled": detector["frames_sampled"],
                "events_triggered": detector["motion_events"] + detector["scene_changes"],
                "motion_events": detector["motion_events"],
                "scene_change_events": detector["scene_changes"],
                "events_discarded": (
                    detector["suppressed_short"] + candidates["discarded"]
                    + candidates["suppressed_duplicates"]
                    + candidates["suppressed_cooldown"]
                ),
                "events_merged": candidates["merged_extensions"],
                "candidates_produced": candidates["candidates_produced"],
                "candidates_retained": candidates["candidates"],
                "candidate_bytes_retained": candidates["disk_bytes"],
                "candidate_bytes_written": candidates["candidate_bytes_written"],
                "old_candidates_expired": candidates["old_candidates_expired"],
                "expired_by_age": candidates["expired_by_age"],
                "expired_by_count": candidates["expired_by_count"],
                "expired_by_bytes": candidates["expired_by_bytes"],
                "missed_buffer": candidates["missed_buffer"],
                "folded_scene_events": candidates["folded_scenes"],
                "detector_error": self._detector_error,
                "transfer": "local_only",
            }

    def health_json(self) -> str:
        status = self.camera_status()
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
                "offline" if not self.running
                else "camera_disconnected" if status["failure_code"] else "online"
            ),
            "failure_code": status["failure_code"],
            "last_captured_at": status["latest_capture_time"],
            "width": None if frame is None else frame.width,
            "height": None if frame is None else frame.height,
            "supported": status["supported"],
            "available": status["available"],
            "capture_active": status["capture_active"],
            "resolution": status["resolution"],
            "latest_capture_time": status["latest_capture_time"],
            "buffer_start_time": status["buffer_start_time"],
            "buffer_end_time": status["buffer_end_time"],
            "buffer_duration": status["buffer_duration"],
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if not self._device_open:
                    self._open_device()
                    if not self._device_open:
                        self._stop.wait(self.retry_interval)
                        continue
                frame = self.source.read()
                self._accept(frame)
            except CaptureError as error:
                self._handle_capture_error(error)
            except Exception:
                self._handle_capture_error(CaptureError("capture_interrupted"))

    def _accept(self, frame: CameraFrame) -> None:
        try:
            captured_at = datetime.fromisoformat(frame.captured_at)
        except ValueError:
            self._note_failure("capture_interrupted")
            return
        if captured_at.tzinfo is None:
            self._note_failure("capture_interrupted")
            return
        try:
            self.buffer.append_frame(
                camera_id=self.source.camera_id,
                captured_at=captured_at,
                width=frame.width,
                height=frame.height,
                payload=frame.jpeg,
                duration_s=self.frame_interval,
            )
        except BufferError as error:
            if error.code == "timestamp_regressed":
                return
            self._note_failure("capture_interrupted")
            return
        with self._lock:
            self._latest = frame
            self._frames += 1
            self._failure_code = None
        self._score(frame, captured_at)

    def _score(self, frame: CameraFrame, captured_at: datetime) -> None:
        """Score one accepted frame. Detector failures stay off the camera status."""
        with self._detector_lock:
            try:
                if frame.sample is None:
                    signals = self.detector.poll(captured_at)
                else:
                    signals = self.detector.observe(
                        captured_at,
                        frame.sample,
                        width=frame.sample_width,
                        height=frame.sample_height,
                    )
                for signal in signals:
                    self.candidates.ingest(signal)
                interval = self.detector.active_interval()
                if interval is None:
                    self.candidates.note_motion(None, None)
                else:
                    self.candidates.note_motion(interval[0], interval[1])
                self.candidates.advance(captured_at)
                self._detector_error = None
            except Exception as error:
                self._detector_error = str(error) or error.__class__.__name__

    def _handle_capture_error(self, error: CaptureError) -> None:
        self._note_failure(error.code)
        if error.code != "encoding_failed":
            self._close_device()
        self._stop.wait(self.retry_interval)

    def _open_device(self) -> None:
        try:
            self.source.open()
        except CaptureError as error:
            self._note_failure(error.code)
            self._close_device()
            return
        except Exception:
            self._note_failure("camera_unavailable")
            self._close_device()
            return
        with self._lock:
            self._device_open = True
            self._failure_code = None

    def _close_device(self) -> None:
        with self._lock:
            was_open = self._device_open
            self._device_open = False
        if not was_open:
            return
        try:
            self.source.close()
        except Exception:
            return

    def _note_failure(self, code: str) -> None:
        with self._lock:
            self._failure_code = code
