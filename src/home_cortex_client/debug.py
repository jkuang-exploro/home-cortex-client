"""Local debug responses for capture and evidence. No Home Cortex call."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from .buffer import BufferError
from .candidates import CandidateFailure
from .evidence import EvidenceFailure


def dispatch(
    runtime: Any, method: str, path: str, query: dict[str, str],
) -> tuple[int, str, dict[str, str], bytes]:
    try:
        return _dispatch(runtime, method, path, query)
    except EvidenceFailure as error:
        status = 404 if error.code == "not_found" else 409
        if error.code == "invalid_duration":
            status = 422
        return _json(status, {"error": {"code": error.code, "message": str(error)}})
    except BufferError as error:
        status = 422 if error.code == "invalid_duration" else 409
        return _json(status, {"error": {"code": error.code, "message": str(error)}})
    except CandidateFailure as error:
        status = 404 if error.code == "not_found" else 409
        return _json(status, {"error": {"code": error.code, "message": str(error)}})


def _dispatch(
    runtime: Any, method: str, path: str, query: dict[str, str],
) -> tuple[int, str, dict[str, str], bytes]:
    if path == "/debug/camera/status" and method == "GET":
        return _json(200, runtime.camera_status())
    if path == "/debug/camera/buffer" and method == "GET":
        return _json(200, {"segments": [segment.as_mapping() for segment in runtime.buffer.segments()]})
    if path == "/debug/camera/latest-frame" and method == "GET":
        return _latest_frame(runtime)
    if path == "/debug/camera/save-last" and method == "POST":
        return _save_last(runtime, query)
    if path == "/debug/evidence/latest" and method in {"GET", "POST"}:
        packaged = runtime.packager.get_latest_still()
        return _json(200, _evidence_body(runtime, packaged.manifest))
    if path == "/debug/evidence/clip" and method == "POST":
        seconds = _seconds(query.get("seconds"))
        packaged = runtime.packager.get_recent_clip(seconds)
        return _json(200, _evidence_body(runtime, packaged.manifest))
    if path == "/debug/detector/status" and method == "GET":
        return _json(200, runtime.detector_status())
    if path == "/debug/candidates" and method == "GET":
        return _json(200, runtime.candidate_index())
    if path == "/debug/events" and method == "GET":
        records = runtime.candidates.store.records()
        return _json(200, {"events": [_event_row(record) for record in records]})
    if path == "/debug/events/stats" and method == "GET":
        return _json(200, runtime.event_stats())
    event_prefix = "/debug/events/"
    if path.startswith(event_prefix) and method == "GET":
        remainder = path[len(event_prefix):]
        if remainder.endswith("/clip"):
            candidate_id = remainder[:-len("/clip")]
            inspected = runtime.candidates.store.inspect(candidate_id)
            if not inspected["evidence_available"]:
                raise CandidateFailure("expired", "candidate clip is no longer available")
            if not inspected["hash_ok"]:
                raise CandidateFailure("invalid_manifest", "candidate clip hash does not match")
            payload = runtime.candidates.store.media(candidate_id)
            return (200, "application/x-home-cortex-clip", {
                "X-Content-Sha256": str(inspected["manifest"]["sha256"]),
                "Cache-Control": "no-store",
            }, payload)
        return _json(200, runtime.candidates.store.inspect(remainder))
    candidate_prefix = "/debug/candidates/"
    if path.startswith(candidate_prefix) and method == "GET":
        candidate_id = path[len(candidate_prefix):]
        if not candidate_id:
            return _json(404, {"error": {"code": "not_found", "message": "unknown debug route"}})
        return _json(200, runtime.candidates.store.inspect(candidate_id))
    if path == "/debug/capabilities" and method == "GET":
        return _json(200, {
            "local_capabilities": runtime.local_capabilities(),
            "session_advertisement": runtime.session_advertisement(),
        })
    if path == "/debug/semantics" and method == "GET":
        return _json(200, runtime.semantic_index())
    semantics_prefix = "/debug/semantics/"
    if path.startswith(semantics_prefix) and method == "GET":
        candidate_id = path[len(semantics_prefix):]
        if not candidate_id:
            return _json(404, {"error": {"code": "not_found", "message": "unknown debug route"}})
        found = runtime.semantic_result(candidate_id)
        if found is None:
            return _json(404, {"error": {"code": "not_found", "message": "semantic result was not found"}})
        return _json(200, found)
    prefix = "/debug/evidence/"
    if path.startswith(prefix) and method == "GET":
        evidence_id = path[len(prefix):]
        if evidence_id in {"", "latest", "clip"}:
            return _json(404, {"error": {"code": "not_found", "message": "unknown debug route"}})
        return _json(200, runtime.packager.store.inspect(evidence_id))
    return _json(404, {"error": {"code": "not_found", "message": "unknown debug route"}})


def _event_row(record: dict[str, object]) -> dict[str, object]:
    started = datetime.fromisoformat(str(record["trigger_started_at"]))
    ended = datetime.fromisoformat(str(record["trigger_ended_at"]))
    return {
        "candidate_id": record["candidate_id"],
        "trigger_type": record["trigger_type"],
        "start": record["trigger_started_at"],
        "end": record["trigger_ended_at"],
        "duration_s": round(max(0.0, (ended - started).total_seconds()), 3),
        "peak_score": record["peak_score"],
        "evidence_id": record["evidence_id"],
        "status": "LOCAL" if record["status"] == "local" else "EXPIRED",
        "transfer_state": record["upload_state"],
    }


def _latest_frame(runtime: Any) -> tuple[int, str, dict[str, str], bytes]:
    segment = runtime.buffer.latest()
    if segment is None:
        raise EvidenceFailure("camera_unavailable", "no captured frame is available")
    status = runtime.camera_status()
    headers = {
        "X-Captured-At": segment.captured_at.isoformat(timespec="milliseconds"),
        "X-Sequence-Number": str(segment.sequence_number),
        "X-Camera-Id": segment.camera_id,
        "X-Fresh": "true" if status.get("available") else "false",
        "X-Content-Sha256": hashlib.sha256(segment.payload).hexdigest(),
        "Cache-Control": "no-store",
    }
    return 200, "image/jpeg", headers, segment.payload


def _save_last(runtime: Any, query: dict[str, str]) -> tuple[int, str, dict[str, str], bytes]:
    seconds = _seconds(query.get("seconds"))
    latest = runtime.buffer.latest()
    if latest is None:
        raise BufferError("buffer_too_short", "buffer has no captured video")
    chosen = runtime.buffer.select_recent(seconds, now=latest.captured_at)
    from .evidence import encode_clip

    payload = encode_clip(chosen)
    headers = {
        "X-Captured-Start": chosen[0].captured_at.isoformat(timespec="milliseconds"),
        "X-Captured-End": chosen[-1].ends_at().isoformat(timespec="milliseconds"),
        "X-Sequence-Start": str(chosen[0].sequence_number),
        "X-Sequence-End": str(chosen[-1].sequence_number),
        "X-Content-Sha256": hashlib.sha256(payload).hexdigest(),
        "Cache-Control": "no-store",
    }
    return 200, "application/x-home-cortex-clip", headers, payload


def _evidence_body(runtime: Any, manifest: dict[str, object]) -> dict[str, object]:
    inspected = runtime.packager.store.inspect(str(manifest["evidence_id"]))
    return {
        "manifest": inspected["manifest"],
        "hash_ok": inspected["hash_ok"],
        "byte_length": inspected["byte_length"],
        "directory": inspected["directory"],
    }


def _seconds(value: str | None) -> float:
    if value is None or not value.strip():
        raise BufferError("invalid_duration", "seconds is required")
    try:
        return float(value)
    except ValueError as error:
        raise BufferError("invalid_duration", "seconds must be a number") from error


def _json(status: int, payload: dict[str, Any]) -> tuple[int, str, dict[str, str], bytes]:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    return status, "application/json", {"Cache-Control": "no-store"}, body
