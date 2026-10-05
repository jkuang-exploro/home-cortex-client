"""Operator commands for the local camera buffer and evidence store."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
import webbrowser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import ClientConfig
from .evidence import EvidenceFailure, decode_clip


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

    detector = commands.add_parser("detector")
    detector_commands = detector.add_subparsers(dest="command", required=True)
    detector_commands.add_parser("status")

    candidates = commands.add_parser("candidates")
    candidate_commands = candidates.add_subparsers(dest="command", required=True)
    candidate_commands.add_parser("list")
    candidate_inspect = candidate_commands.add_parser("inspect")
    candidate_inspect.add_argument("candidate_id")

    events = commands.add_parser("events", help="Inspect local visual-change candidates")
    event_commands = events.add_subparsers(dest="command", required=True)
    event_commands.add_parser("list")
    event_commands.add_parser("stats")
    event_show = event_commands.add_parser("show")
    event_show.add_argument("candidate_id")
    event_clip = event_commands.add_parser("clip")
    event_clip.add_argument("candidate_id")
    event_clip.add_argument("--output", type=Path, help="Directory for clip and browser player")
    event_clip.add_argument("--open", action="store_true", help="Open the local player")

    semantics = commands.add_parser("semantics", help="Inspect local semantic filter results")
    semantic_commands = semantics.add_subparsers(dest="command", required=True)
    semantic_commands.add_parser("list")
    semantic_show = semantic_commands.add_parser("show")
    semantic_show.add_argument("candidate_id")

    capabilities = commands.add_parser("capabilities", help="Show local capability declarations")
    capability_commands = capabilities.add_subparsers(dest="command", required=True)
    capability_commands.add_parser("show")

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
        elif args.group == "detector" and args.command == "status":
            print(_body(_get(base_url, "/debug/detector/status")))
        elif args.group == "candidates" and args.command == "list":
            print(_body(_get(base_url, "/debug/candidates")))
        elif args.group == "candidates" and args.command == "inspect":
            print(_body(_get(base_url, "/debug/candidates/" + args.candidate_id)))
        elif args.group == "events" and args.command == "list":
            print(_body(_get(base_url, "/debug/events")))
        elif args.group == "events" and args.command == "stats":
            print(_body(_get(base_url, "/debug/events/stats")))
        elif args.group == "events" and args.command == "show":
            print(_body(_get(base_url, "/debug/events/" + args.candidate_id)))
        elif args.group == "events" and args.command == "clip":
            print(json.dumps(_export_event_clip(
                base_url, args.candidate_id, args.output, open_player=args.open,
            ), indent=2, sort_keys=True))
        elif args.group == "semantics" and args.command == "list":
            print(_body(_get(base_url, "/debug/semantics")))
        elif args.group == "semantics" and args.command == "show":
            print(_body(_get(base_url, "/debug/semantics/" + args.candidate_id)))
        elif args.group == "capabilities" and args.command == "show":
            print(_body(_get(base_url, "/debug/capabilities")))
        else:
            parser.error("unknown command")
    except (HTTPError, URLError, OSError, RuntimeError, EvidenceFailure, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


def _export_event_clip(
    base_url: str, candidate_id: str, output: Path | None, *, open_player: bool,
) -> dict[str, object]:
    detail = json.loads(_body(_get(base_url, "/debug/events/" + candidate_id)))
    if not detail.get("evidence_available") or not detail.get("hash_ok"):
        raise RuntimeError("candidate clip is unavailable or failed its hash check")
    manifest = detail["manifest"]
    status, _headers, payload = _get(base_url, "/debug/events/" + candidate_id + "/clip")
    _raise_for_status(status, payload)
    if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
        raise RuntimeError("downloaded candidate clip failed its hash check")
    frames = decode_clip(payload)
    root = output or Path.cwd() / ("candidate-" + candidate_id.removeprefix("candidate:"))
    root.mkdir(parents=True, exist_ok=True)
    (root / "clip.hcc").write_bytes(payload)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    entries = []
    for index, (sequence, captured_ms, jpeg) in enumerate(frames):
        name = f"frame-{index:06d}.jpg"
        (root / name).write_bytes(jpeg)
        entries.append({"file": name, "sequence": sequence, "captured_ms": captured_ms})
    player = root / "index.html"
    player.write_text(_event_player(candidate_id, entries, manifest), encoding="utf-8")
    if open_player:
        webbrowser.open(player.resolve().as_uri())
    return {
        "candidate_id": candidate_id,
        "frames": len(entries),
        "hash_ok": True,
        "clip": str((root / "clip.hcc").resolve()),
        "player": str(player.resolve()),
        "transfer_state": detail["candidate"]["upload_state"],
        "has_left_client": detail["provenance"]["transfer"]["has_left_client"],
    }


def _event_player(
    candidate_id: str, frames: list[dict[str, object]], manifest: dict[str, object],
) -> str:
    encoded = json.dumps(frames, separators=(",", ":"))
    shown_manifest = html.escape(json.dumps(manifest, indent=2, sort_keys=True))
    return (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{html.escape(candidate_id)}</title>"
        "<style>body{font:15px system-ui;background:#111;color:#eee;margin:2rem}"
        "img{display:block;max-width:100%;max-height:70vh;margin:1rem 0}"
        "input{width:min(90vw,900px)}pre{white-space:pre-wrap}</style>"
        f"<h1>{html.escape(candidate_id)}</h1>"
        "<p>Local candidate evidence. No automatic Home Cortex transfer.</p>"
        "<button id='play'>Play</button><input id='seek' type='range' min='0' "
        f"max='{max(0, len(frames) - 1)}' value='0'>"
        "<span id='position'></span><img id='frame' alt='candidate frame'>"
        f"<details><summary>Evidence manifest</summary><pre>{shown_manifest}</pre></details>"
        f"<script>const frames={encoded};let index=0,timer=null;"
        "const image=document.getElementById('frame'),seek=document.getElementById('seek'),"
        "position=document.getElementById('position'),play=document.getElementById('play');"
        "function show(i){index=i;image.src=frames[i].file;seek.value=i;"
        "position.textContent=`${i+1}/${frames.length} · sequence ${frames[i].sequence} "
        "· ${new Date(frames[i].captured_ms).toISOString()}`;}"
        "function stop(){if(timer!==null)clearTimeout(timer);timer=null;play.textContent='Play';}"
        "function step(){if(index+1>=frames.length){stop();return;}"
        "const delay=Math.max(40,Math.min(1000,frames[index+1].captured_ms-frames[index].captured_ms));"
        "timer=setTimeout(()=>{show(index+1);step();},delay);}"
        "play.onclick=()=>{if(timer!==null){stop();return;}if(index+1>=frames.length)show(0);"
        "play.textContent='Pause';step();};"
        "seek.oninput=()=>{stop();show(Number(seek.value));};if(frames.length)show(0);</script>"
    )


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
