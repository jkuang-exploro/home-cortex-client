"""Fulfill one polled Home Cortex observation without choosing when to observe."""
from __future__ import annotations

from .backend import BackendSession
from .evidence import EvidenceFailure, EvidencePackager


_TRANSPORT_CODES = {
    "camera_unavailable": "camera_unavailable",
    "camera_permission_denied": "camera_unavailable",
    "camera_busy": "camera_unavailable",
    "capture_interrupted": "capture_failure",
    "encoding_failed": "capture_failure",
    "capture_failure": "capture_failure",
    "evidence_stale": "evidence_stale",
    "buffer_too_short": "buffer_too_short",
    "invalid_duration": "invalid_duration",
}


def fulfill_pending(session: BackendSession, packager: EvidencePackager) -> int:
    """Package and return every outstanding explicit observation command."""
    fulfilled = 0
    for command in session.poll_commands():
        command_id = command.get("command_id")
        if not isinstance(command_id, str) or not command_id:
            continue
        fulfilled += 1
        try:
            packaged = _select(packager, command)
        except EvidenceFailure as error:
            session.submit_observation_failure(
                command_id, _TRANSPORT_CODES.get(error.code, "capture_failure"), str(error),
            )
            continue
        except Exception:
            session.submit_observation_failure(command_id, "capture_failure", "capture failed")
            continue
        session.submit_observation(command_id, packaged.manifest, packaged.payload)
    return fulfilled


def _select(packager: EvidencePackager, command: dict):
    operation = command.get("operation")
    if operation == "vision.observe":
        return packager.get_latest_still()
    if operation == "vision.observe_clip":
        return packager.get_recent_clip(command.get("duration_seconds"))
    raise EvidenceFailure("capture_failure", "unsupported observation operation")
