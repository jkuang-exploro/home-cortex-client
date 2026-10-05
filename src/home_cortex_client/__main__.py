"""Run the Home Cortex edge client."""
from __future__ import annotations

import argparse
import sys
import threading
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
    if args and args[0] in {
        "camera", "evidence", "detector", "candidates", "events", "semantics", "capabilities",
    }:
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
        detection=settings.detection,
        clock=wall_clock,
    )
    runtime.set_session_capabilities(settings.session_capabilities)
    observation = threading.Event()
    analyzer = _start_analyzer(runtime, settings, observation, wall_clock)
    publisher = _start_publisher(runtime, analyzer, session, settings, wall_clock)
    endpoint = runtime.start()
    if analyzer is not None:
        analyzer.start()
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
    print("  candidates=local_only")
    print(f"  analyzer={settings.analyzer}")
    print(f"  promotion={'queued' if publisher is not None else 'off'}")
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
                # Local semantic filtering waits while an explicit observation is packaged.
                observation.set()
                try:
                    try:
                        fulfill_pending(session, runtime.packager)
                    except Exception as error:
                        print(f"Home Cortex observation poll failed: {error}", file=sys.stderr)
                finally:
                    observation.clear()
                if publisher is not None and advertised and "vision.autonomous_promotion" in advertised:
                    try:
                        publisher.recover(analyzer.store)
                        publisher.flush(session.publish_evidence)
                    except Exception as error:
                        print(f"Home Cortex promotion flush failed: {error}", file=sys.stderr)
                next_poll = time.monotonic() + 0.5
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if session is not None:
            try:
                session.disconnect()
            except Exception as error:
                print(f"Home Cortex disconnect failed: {error}", file=sys.stderr)
        if analyzer is not None:
            analyzer.stop()
        runtime.stop()
    return 0


def _start_analyzer(runtime: EdgeRuntime, settings: ClientConfig, hold: threading.Event, clock):
    """Start no model unless the operator selects one. The default leaves Stage 2 unchanged."""
    if settings.analyzer == "off":
        return None
    from .analyze import SemanticWorker
    from .perception import open_model
    from .policy import policy_config

    root = runtime.candidates.store.root.with_name(runtime.candidates.store.root.name + "-semantics")
    worker = SemanticWorker(
        runtime.candidates.store,
        open_model(settings.analyzer),
        root=root,
        hold=hold,
        clock=clock,
        policy=policy_config(
            repeat_window_s=settings.promotion_window_s,
            minimum_promote_confidence=settings.promotion_min_confidence,
            repeat=settings.promotion_repeat,
        ),
        override=settings.promotion_override,
    )
    runtime.attach_analyzer(worker)
    return worker


def _start_publisher(runtime: EdgeRuntime, analyzer, session, settings: ClientConfig, clock):
    """Queue PROMOTE evidence only while a Home Cortex session exists."""
    if analyzer is None or session is None:
        return None
    from .publish import PromotionOutbox

    publisher = PromotionOutbox(
        runtime.candidates.store,
        max_pending=settings.promotion_queue_limit,
        max_age_s=settings.promotion_max_age_s,
        clock=clock,
        client_runtime_version="home-cortex-client",
    )
    analyzer.set_promoted_listener(publisher.enqueue)
    publisher.recover(analyzer.store)
    return publisher


def _advertised(runtime: EdgeRuntime) -> list[str]:
    return runtime.session_advertisement()


def _wait_for_first_sample(runtime: EdgeRuntime) -> None:
    deadline = time.monotonic() + min(2.0, runtime.freshness_s)
    while time.monotonic() < deadline:
        status = runtime.camera_status()
        if status["available"] or status["failure_code"]:
            return
        runtime.wait(0.05)


if __name__ == "__main__":
    sys.exit(main())
