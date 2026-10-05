"""Camera-frame contract local to the edge runtime."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class CameraFrame:
    """One captured frame. Camera-native buffers stay inside the source.

    ``sample`` is an optional downsampled grayscale grid for the local
    visual-change detector. Capture succeeds when the sample is absent.
    """

    captured_at: str
    width: int
    height: int
    jpeg: bytes
    sample: bytes | None = None
    sample_width: int = 0
    sample_height: int = 0


class CameraSource(Protocol):
    device_id: str
    camera_id: str

    def open(self) -> None: ...

    def read(self) -> CameraFrame: ...

    def close(self) -> None: ...


def capture_timestamp(when: datetime | None = None) -> str:
    """Return timezone-aware ISO-8601 for the serialized observation seam."""
    moment = when if when is not None else datetime.now().astimezone()
    return moment.isoformat(timespec="milliseconds")

