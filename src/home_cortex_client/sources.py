"""Replaceable camera sources. Mac-specific capture stays in MacCameraSource."""
from __future__ import annotations

import platform
from datetime import datetime, timedelta, timezone

from .capture import CaptureError, classify_camera_failure
from .frames import CameraFrame, capture_timestamp


DEFAULT_DEVICE_ID = "device:dev_macbook"
DEFAULT_CAMERA_ID = "camera:built_in"

# 1x1 JPEG; synthetic tests only need a valid encoded payload, not scene content.
TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000"
    "ffdb0043000101010101010101010101010101010101010101010101010101"
    "01010101010101010101010101010101010101010101010101010101010101"
    "01010101010101ffc0000b080001000101011100ffc4001400010000000000"
    "0000000000000000000003ffda000800010001003f00d2cf20ffd9"
)


class SyntheticCameraSource:
    """Hardware-free source for tests and local development."""

    def __init__(
        self,
        *,
        device_id: str = DEFAULT_DEVICE_ID,
        camera_id: str = DEFAULT_CAMERA_ID,
        width: int = 640,
        height: int = 480,
        fps: float = 10.0,
        jpeg: bytes = TINY_JPEG,
        start: datetime | None = None,
        clock=None,
    ) -> None:
        self.device_id = device_id
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self._jpeg = jpeg
        self._clock = clock
        self._start = start or datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)
        self._index = 0
        self._opened = False

    def open(self) -> None:
        self._opened = True
        self._index = 0

    def read(self) -> CameraFrame:
        if not self._opened:
            raise RuntimeError("SyntheticCameraSource is closed")
        when = self._clock() if self._clock is not None else (
            self._start + timedelta(seconds=self._index / self.fps)
        )
        self._index += 1
        return CameraFrame(
            captured_at=capture_timestamp(when),
            width=self.width,
            height=self.height,
            jpeg=self._jpeg,
        )

    def close(self) -> None:
        self._opened = False


class MacCameraSource:
    """Built-in camera via OpenCV. Swap this adapter for MicroDuck later."""

    def __init__(
        self,
        *,
        index: int = 0,
        device_id: str = DEFAULT_DEVICE_ID,
        camera_id: str = DEFAULT_CAMERA_ID,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        jpeg_quality: int = 80,
    ) -> None:
        self.index = index
        self.device_id = device_id
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self._cv2 = None
        self._capture = None

    def open(self) -> None:
        try:
            cv2 = _load_cv2()
        except RuntimeError as error:
            raise CaptureError("camera_unavailable") from error
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
        try:
            capture = cv2.VideoCapture(self.index, backend)
        except Exception as error:
            raise classify_camera_failure(error) from error
        if not capture.isOpened():
            capture.release()
            raise CaptureError("camera_unavailable")
        if self.width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._cv2 = cv2
        self._capture = capture

    def read(self) -> CameraFrame:
        if self._capture is None or self._cv2 is None:
            raise CaptureError("camera_unavailable")
        try:
            ok, image = self._capture.read()
        except Exception as error:
            raise classify_camera_failure(error) from error
        if not ok or image is None:
            raise CaptureError("capture_interrupted")
        captured_at = capture_timestamp()
        try:
            height, width = image.shape[:2]
        except Exception as error:
            raise CaptureError("capture_interrupted") from error
        try:
            ok, encoded = self._cv2.imencode(
                ".jpg", image, [int(self._cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
        except Exception as error:
            raise CaptureError("encoding_failed") from error
        if not ok:
            raise CaptureError("encoding_failed")
        return CameraFrame(
            captured_at=captured_at,
            width=int(width),
            height=int(height),
            jpeg=encoded.tobytes(),
        )

    def close(self) -> None:
        capture = self._capture
        self._capture = None
        self._cv2 = None
        if capture is not None:
            capture.release()


def _load_cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python is not installed") from error
    return cv2

