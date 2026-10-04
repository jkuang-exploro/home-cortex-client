"""Run the Home Cortex edge client."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

from .backend import BackendSession
from .commands import fulfill_pending
from .config import ClientConfig
from .runtime import EdgeRuntime
from .sources import MacCameraSource, SyntheticCameraSource
from .stream import StreamConfig


def build_parser(config: ClientConfig | None = None) -> argparse.ArgumentParser:
    defaults = config or ClientConfig.from_env()
    parser = argparse.ArgumentParser(
        description="Home Cortex edge-device camera and live-stream runtime."
    )
    parser.add_argument(
        "--source",
        choices=("mac", "synthetic"),
        default=defaults.source,
        help="mac uses a local camera; synthetic needs no hardware",
    )
    parser.add_argument("--host", default=defaults.stream_host)
    parser.add_argument("--port", type=int, default=defaults.stream_port)
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--height", type=int, default=defaults.height)
    parser.add_argument("--fps", type=float, default=defaults.fps)
    parser.add_argument("--device-id", default=defaults.device_id)
    parser.add_argument("--camera-id", default=defaults.camera_id)
    parser.add_argument("--embodiment-id", default=defaults.embodiment_id)
    parser.add_argument("--cortex-url", default=defaults.cortex_url)
    parser.add_argument("--index", type=int, default=defaults.camera_index)
    parser.add_argument("--buffer-seconds", type=float, default=defaults.buffer_seconds)
    parser.add_argument("--freshness-seconds", type=float, default=defaults.freshness_seconds)
    parser.add_argument("--evidence-dir", default=defaults.evidence_dir)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"camera", "evidence"}:
        from .debug_cli import main as debug_main
        return debug_main(args)
    return run(args)


def run(argv: list[str]) -> int:
    settings = ClientConfig.from_env()
    args = build_parser(settings).parse_args(argv)
    if bool(args.cortex_url) != bool(settings.cortex_api_key):
        raise SystemExit("Set both Home Cortex URL and HOME_CORTEX_CLIENT_CORTEX_API_KEY")
    session = (BackendSession(args.cortex_url, settings.cortex_api_key, args.embodiment_id)
               if args.cortex_url else None)
    config = StreamConfig(host=args.host, port=args.port)
    wall_clock = lambda: datetime.now().astimezone()
    if args.source == "synthetic":
        source = SyntheticCameraSource(
            device_id=args.device_id,
            camera_id=args.camera_id,
            width=args.width or 640,
            height=args.height or 480,
            fps=args.fps,
            clock=wall_clock,
        )
    else:
        source = MacCameraSource(
            index=args.index,
            device_id=args.device_id,
            camera_id=args.camera_id,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
    runtime = EdgeRuntime(
        source,
        config=config,
        fps=args.fps,
        buffer_seconds=args.buffer_seconds,
        freshness_seconds=args.freshness_seconds,
        evidence_dir=args.evidence_dir,
        embodiment_id=args.embodiment_id,
        clock=wall_clock,
    )
    endpoint = runtime.start()
    advertised: list[str] | None = None
    try:
        if session is not None:
            _wait_for_first_sample(runtime)
            advertised = _advertised(runtime)
            session.connect(advertised)
    except Exception:
        runtime.stop()
        raise
    print("Home Cortex Client")
    print(f"  device_id={source.device_id}")
    print(f"  camera_id={source.camera_id}")
    print(f"  embodiment_id={args.embodiment_id}")
    print(f"  source={args.source}")
    print(f"  transport={config.transport}")
    print(f"  stream={endpoint}")
    print(f"  viewer={runtime.viewer}")
    print(f"  buffer_seconds={args.buffer_seconds}")
    print(f"  Home Cortex session={'online' if session else 'disabled'}")
    print("  stop=Ctrl-C")
    try:
        next_heartbeat = time.monotonic() + 10
        next_poll = time.monotonic()
        while runtime.running:
            runtime.wait(0.2)
            if session is None:
                continue
            current = _advertised(runtime)
            if current != advertised:
                try:
                    session.update_capabilities(current)
                    advertised = current
                except Exception as error:
                    print(f"Home Cortex capability update failed: {error}", file=sys.stderr)
            now = time.monotonic()
            if now >= next_heartbeat:
                try:
                    session.heartbeat()
                except Exception as error:
                    print(f"Home Cortex heartbeat failed: {error}", file=sys.stderr)
                next_heartbeat = time.monotonic() + 10
            if now >= next_poll:
                try:
                    fulfill_pending(session, runtime.packager)
                except Exception as error:
                    print(f"Home Cortex observation poll failed: {error}", file=sys.stderr)
                next_poll = time.monotonic() + 0.5
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if session is not None:
            try:
                session.disconnect()
            except Exception as error:
                print(f"Home Cortex disconnect failed: {error}", file=sys.stderr)
        runtime.stop()
    return 0


def _advertised(runtime: EdgeRuntime) -> list[str]:
    return ["vision.observe"] if runtime.camera_status()["available"] else []


def _wait_for_first_sample(runtime: EdgeRuntime) -> None:
    deadline = time.monotonic() + min(2.0, runtime.freshness_s)
    while time.monotonic() < deadline:
        status = runtime.camera_status()
        if status["available"] or status["failure_code"]:
            return
        runtime.wait(0.05)


if __name__ == "__main__":
    sys.exit(main())
