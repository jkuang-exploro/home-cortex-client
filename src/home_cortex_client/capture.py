"""Stable capture failures. Native camera text stays inside the source."""
from __future__ import annotations


CAPTURE_CODES = frozenset({
    "camera_permission_denied",
    "camera_unavailable",
    "camera_busy",
    "capture_interrupted",
    "encoding_failed",
})


class CaptureError(Exception):
    """A classified camera failure that must not stop the client process."""

    def __init__(self, code: str) -> None:
        if code not in CAPTURE_CODES:
            raise ValueError(f"unknown capture code: {code}")
        self.code = code
        super().__init__(code)


def classify_camera_failure(error: BaseException) -> CaptureError:
    """Map a vendor exception to a stable code without retaining its text."""
    text = str(error).casefold()
    if any(word in text for word in ("authoriz", "permission", "not allowed", "privacy")):
        return CaptureError("camera_permission_denied")
    if any(word in text for word in ("busy", "in use", "already open")):
        return CaptureError("camera_busy")
    return CaptureError("camera_unavailable")
