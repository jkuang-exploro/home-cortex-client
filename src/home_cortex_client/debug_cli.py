"""Operator commands for the local camera buffer and evidence store."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import ClientConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="home-cortex-client", description="Local camera and evidence checks.")
    parser.add_argument("--base-url", help="Running client base URL. Defaults to the configured stream address.")
    commands = parser.add_subparsers(dest="group", required=True)

    camera = commands.add_parser("camera")
    camera_commands = camera.add_subparsers(dest="command", required=True)
    camera_commands.add_parser("status")
    latest = camera_commands.add_parser("latest-frame")
    latest.add_argument("--output", type=Path, default=Path("latest-frame.jpg"))
    camera_commands.add_parser("list-buffer")
    save_last = camera_commands.add_parser("save-last")
    save_last.add_argument("seconds", type=float)
    save_last.add_argument("--output", type=Path)

    evidence = commands.add_parser("evidence")
    evidence_commands = evidence.add_subparsers(dest="command", required=True)
    evidence_commands.add_parser("latest")
    clip = evidence_commands.add_parser("clip")
    clip.add_argument("--seconds", type=float, required=True)
    inspect = evidence_commands.add_parser("inspect")
    inspect.add_argument("evidence_id")

    args = parser.parse_args(argv)
    base_url = (args.base_url or _default_base_url()).rstrip("/")
    try:
        if args.group == "camera" and args.command == "status":
            print(_body(_get(base_url, "/debug/camera/status")))
        elif args.group == "camera" and args.command == "latest-frame":
            status, headers, body = _request(base_url, "GET", "/debug/camera/latest-frame")
            _raise_for_status(status, body)
            args.output.write_bytes(body)
            print(json.dumps({
                "output": str(args.output),
                "captured_at": headers.get("X-Captured-At"),
                "sequence_number": headers.get("X-Sequence-Number"),
                "fresh": headers.get("X-Fresh"),
                "sha256": headers.get("X-Content-Sha256"),
                "bytes": len(body),
            }, indent=2, sort_keys=True))
        elif args.group == "camera" and args.command == "list-buffer":
            print(_body(_get(base_url, "/debug/camera/buffer")))
        elif args.group == "camera" and args.command == "save-last":
            output = args.output or Path(f"last-{args.seconds:g}s.hcc")
            status, headers, body = _request(
                base_url, "POST", f"/debug/camera/save-last?seconds={args.seconds}",
            )
            _raise_for_status(status, body)
            output.write_bytes(body)
            print(json.dumps({
                "output": str(output),
                "captured_start": headers.get("X-Captured-Start"),
                "captured_end": headers.get("X-Captured-End"),
                "sequence_start": headers.get("X-Sequence-Start"),
                "sequence_end": headers.get("X-Sequence-End"),
                "sha256": headers.get("X-Content-Sha256"),
                "bytes": len(body),
            }, indent=2, sort_keys=True))
        elif args.group == "evidence" and args.command == "latest":
            print(_body(_get(base_url, "/debug/evidence/latest")))
        elif args.group == "evidence" and args.command == "clip":
            print(_body(_request(
                base_url, "POST", f"/debug/evidence/clip?seconds={args.seconds}",
            )))
        elif args.group == "evidence" and args.command == "inspect":
            print(_body(_get(base_url, "/debug/evidence/" + args.evidence_id)))
        else:
            parser.error("unknown command")
    except (HTTPError, URLError, OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


def _default_base_url() -> str:
    config = ClientConfig.from_env()
    return f"http://{config.stream_host}:{config.stream_port}"


def _get(base_url: str, path: str) -> tuple[int, dict[str, str], bytes]:
    return _request(base_url, "GET", path)


def _request(base_url: str, method: str, path: str) -> tuple[int, dict[str, str], bytes]:
    request = Request(base_url + path, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, dict(response.headers), response.read()
    except HTTPError as error:
        body = error.read()
        _raise_for_status(error.code, body)
        raise


def _body(result: tuple[int, dict[str, str], bytes]) -> str:
    status, _headers, body = result
    _raise_for_status(status, body)
    return body.decode("utf-8")


def _raise_for_status(status: int, body: bytes) -> None:
    if status >= 400:
        text = body.decode("utf-8", errors="replace").strip()
        raise RuntimeError(text or f"client debug request failed ({status})")


if __name__ == "__main__":
    sys.exit(main())
