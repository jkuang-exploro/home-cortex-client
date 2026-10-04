"""Device-local configuration for the edge runtime."""
from __future__ import annotations

from dataclasses import dataclass
import os


ENV_PREFIX = "HOME_CORTEX_CLIENT_"


@dataclass(frozen=True)
class ClientConfig:
    source: str = "mac"
    embodiment_id: str = "embodiment:macbook-0"
    cortex_url: str | None = None
    cortex_api_key: str | None = None
    device_id: str = "device:dev_macbook"
    camera_id: str = "camera:built_in"
    camera_index: int = 0
    stream_host: str = "127.0.0.1"
    stream_port: int = 8088
    width: int | None = None
    height: int | None = None
    fps: float = 10.0
    buffer_seconds: float = 60.0
    freshness_seconds: float = 2.0
    evidence_dir: str | None = None
    evidence_max_items: int = 8

    @classmethod
    def from_env(cls) -> "ClientConfig":
        return cls(
            source=_text("SOURCE", cls.source),
            embodiment_id=_text("EMBODIMENT_ID", cls.embodiment_id),
            cortex_url=_value("CORTEX_URL"),
            cortex_api_key=_value("CORTEX_API_KEY"),
            device_id=_text("DEVICE_ID", cls.device_id),
            camera_id=_text("CAMERA_ID", cls.camera_id),
            camera_index=_integer("CAMERA_INDEX", cls.camera_index),
            stream_host=_text("STREAM_HOST", cls.stream_host),
            stream_port=_integer("STREAM_PORT", cls.stream_port),
            width=_optional_integer("WIDTH"),
            height=_optional_integer("HEIGHT"),
            fps=_number("FPS", cls.fps),
            buffer_seconds=_number("BUFFER_SECONDS", cls.buffer_seconds),
            freshness_seconds=_number("FRESHNESS_SECONDS", cls.freshness_seconds),
            evidence_dir=_value("EVIDENCE_DIR"),
            evidence_max_items=_integer("EVIDENCE_MAX_ITEMS", cls.evidence_max_items),
        )


def _value(name: str) -> str | None:
    value = os.environ.get(f"{ENV_PREFIX}{name}")
    if value is None or not value.strip():
        return None
    return value.strip()


def _text(name: str, default: str) -> str:
    return _value(name) or default


def _integer(name: str, default: int) -> int:
    value = _value(name)
    return default if value is None else int(value)


def _optional_integer(name: str) -> int | None:
    value = _value(name)
    return None if value is None else int(value)


def _number(name: str, default: float) -> float:
    value = _value(name)
    return default if value is None else float(value)
