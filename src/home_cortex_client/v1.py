"""MacBook's V1 transport adapter. Local perception remains in its existing owners."""
from __future__ import annotations

import base64
import platform
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode

from .credentials import Credentials, HTTPTransport
from .evidence import EvidenceFailure, PackagedEvidence
from .protocol import (
    MAX_MEDIA_BYTES,
    ProtocolError,
    canonical,
    closed,
    common,
    identifier,
    integer,
    request,
    response,
    text,
    timestamp,
    validate_request,
    validate_response,
)
from .receipts import ReceiptJournal

MESSAGES = "/client-interface/v1/messages"
VISION = frozenset({"vision.observe", "vision.observe_clip", "vision.autonomous_promotion"})
LOCAL_FAILURES = {
    "camera_unavailable": ("TEMPORARILY_UNAVAILABLE", "Camera capture is unavailable."),
    "camera_permission_denied": ("PERMISSION_DENIED", "Local camera permission is unavailable."),
    "camera_busy": ("BUSY", "Local camera is busy."),
    "buffer_too_short": ("TEMPORARILY_UNAVAILABLE", "The exact recent clip interval is unavailable."),
    "evidence_stale": ("TEMPORARILY_UNAVAILABLE", "Fresh evidence is unavailable."),
    "invalid_duration": ("INVALID_ARGUMENT", "The requested duration is invalid."),
    "capture_interrupted": ("TEMPORARILY_UNAVAILABLE", "Camera capture was interrupted."),
    "encoding_failed": ("INTERNAL_ERROR", "Evidence encoding failed."),
    "capture_failure": ("INTERNAL_ERROR", "Evidence capture failed."),
    "semantic_unavailable": ("TEMPORARILY_UNAVAILABLE", "Local promotion analysis is unavailable."),
}


class ObservationPackager(Protocol):
    def get_v1_still(self, sent_at: datetime) -> PackagedEvidence: ...
    def get_v1_clip(self, duration_ms: int) -> PackagedEvidence: ...


def local_error(code: str) -> ProtocolError:
    canonical_code, message = LOCAL_FAILURES.get(code, ("INTERNAL_ERROR", "Observation failed."))
    return ProtocolError(canonical_code, code if code in LOCAL_FAILURES else "capture_failure", message,
                         retryable=canonical_code in {"TEMPORARILY_UNAVAILABLE", "BUSY"})


def capability_manifest(runtime: Any, configured: tuple[str, ...], revision: int = 1) -> dict[str, Any]:
    """Presence means implemented support; camera failure does not remove support."""
    camera = runtime.camera_status()
    local = runtime.local_capabilities()
    caps: list[dict[str, Any]] = []
    for name in configured:
        if name not in VISION:
            raise ValueError("Only implemented vision capabilities may be advertised")
        available = bool(local.get(name))
        if name == "vision.observe_clip":
            available = available and camera["buffer_duration"] > 0
        item: dict[str, Any] = {"name": name, "schema_version": 1,
                    "availability": "AVAILABLE" if available else "TEMPORARILY_UNAVAILABLE"}
        if not available:
            unavailable = "buffer_too_short" if name == "vision.observe_clip" else "camera_unavailable"
            if name == "vision.autonomous_promotion" and not local.get("vision.semantic_filter"):
                unavailable = "semantic_unavailable"
            item["reason"] = local_error(camera.get("failure_code") or unavailable).as_mapping()
        if name == "vision.observe_clip":
            item["limits"] = {"max_clip_duration_ms": min(60000, max(1, int(runtime.buffer.duration_s * 1000)))}
        caps.append(item)
    return {"revision": revision, "capabilities": caps}


class V1Session:
    def __init__(self, credentials: Credentials, state_dir: Path, *, transport: Any | None = None,
                 endpoint: str | None = None, clock: Callable[[], datetime] | None = None,
                 monotonic: Callable[[], float] = time.monotonic):
        self.credentials = credentials
        self.transport = transport or HTTPTransport(endpoint or credentials.server_endpoint, credentials.context())
        self.clock = clock or (lambda: datetime.now(UTC))
        self.monotonic = monotonic
        self.journal = ReceiptJournal(state_dir, identity=f"{credentials.client_id}\n{credentials.embodiment_id}")
        self._lock = threading.RLock()
        self.session_id: str | None = None
        self.state = "DISCONNECTED"
        self._lease_until = 0.0
        self._next_heartbeat = 0.0
        self._manifest: dict[str, Any] | None = None
        self._effective: frozenset[str] = frozenset()
        self.cursor: str | None = None
        self._polled_cursor: str | None = None
        self._pending: dict[str, dict[str, Any]] = {}
        self._event_sequences: dict[str, int] = {}
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self.last_error: str | None = None
        self.auth_failed = False
        self.promotion_max_age_s = 6 * 3600.0
        self.max_media_bytes = MAX_MEDIA_BYTES

    @property
    def active(self) -> bool:
        with self._lock:
            if self.state == "ACTIVE" and self.monotonic() >= self._lease_until:
                self.state = "STALE"
                self._effective = frozenset()
            return self.state == "ACTIVE"

    @property
    def effective_capabilities(self) -> frozenset[str]:
        return self._effective if self.active else frozenset()

    def _target(self, *, registration: bool = False) -> dict[str, Any]:
        return {"embodiment_id": self.credentials.embodiment_id,
                    "session_id": None if registration else self.session_id}

    def _control(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            target = self._target(registration=operation == "session.register")
            key = canonical({"operation": operation, "target": target, "arguments": arguments})
            original = self._pending.get(key)
            if original is None or self.clock() >= timestamp(original["deadline_at"]):
                original = request(operation, arguments, target, self.clock())
                self._pending[key] = original
        started = self.monotonic()
        reply = self.transport.request("POST", MESSAGES, original, timeout=5)
        result = validate_response(reply, original)
        self._accept_view(result, original, self.monotonic() - started)
        with self._lock:
            self._pending.pop(key, None)
        return result

    def _accept_view(self, view: Any, original: dict[str, Any], round_trip: float) -> None:
        item = closed(view, {"client_id", "embodiment_id", "session_id", "state", "connected_at", "last_seen_at",
                             "server_time", "lease_expires_at", "heartbeat_interval_ms", "lease_duration_ms",
                             "manifest_revision", "effective_capabilities"})
        if item["client_id"] != self.credentials.client_id or item["embodiment_id"] != self.credentials.embodiment_id:
            raise ProtocolError("PERMISSION_DENIED", "wrong_identity", "V1 session identity does not match credentials.")
        if not text(item["session_id"]).startswith("runtime-session:"):
            raise ProtocolError("INVALID_ARGUMENT", "invalid_session", "Invalid V1 session identity.")
        if original["operation"] != "session.register" and item["session_id"] != original["target"]["session_id"]:
            raise ProtocolError("CONFLICT", "stale_session", "V1 session fence changed.")
        if item["state"] not in {"ACTIVE", "STALE", "EXPIRED", "REPLACED", "DISCONNECTED"}:
            raise ProtocolError("INVALID_ARGUMENT", "invalid_session", "Invalid V1 session state.")
        heartbeat = integer(item["heartbeat_interval_ms"], 1000)
        lease = integer(item["lease_duration_ms"], heartbeat * 3)
        integer(item["manifest_revision"], 1)
        if not isinstance(item["effective_capabilities"], list) or any(not isinstance(name, str) or name not in VISION for name in item["effective_capabilities"]):
            raise ProtocolError("INVALID_ARGUMENT", "invalid_capabilities", "Unexpected effective capability.")
        manifest = original["arguments"].get("manifest", self._manifest)
        if manifest is not None:
            advertised = {cap["name"] for cap in manifest["capabilities"] if cap["availability"] == "AVAILABLE"}
            effective = item["effective_capabilities"]
            if len(set(effective)) != len(effective) or set(effective) - advertised:
                raise ProtocolError("INVALID_ARGUMENT", "invalid_capabilities", "Unexpected effective capability.")
        for key in ("connected_at", "last_seen_at", "server_time", "lease_expires_at"):
            timestamp(item[key])
        remaining = (timestamp(item["lease_expires_at"]) - timestamp(item["server_time"])).total_seconds()
        if remaining > lease / 1000 + 0.001:
            raise ProtocolError("INVALID_ARGUMENT", "invalid_lease", "Invalid V1 lease duration.")
        with self._lock:
            now = self.monotonic()
            self.session_id = item["session_id"]
            self.state = item["state"]
            wall_remaining = (timestamp(item["lease_expires_at"]) - self.clock()).total_seconds()
            budget = max(0, min(remaining - round_trip - 2.0, wall_remaining - 2.0))
            self._lease_until = now + budget
            self._next_heartbeat = now + min(heartbeat / 1000, budget / 2)
            self._effective = frozenset(item["effective_capabilities"])
            _ = self.active  # apply already-expired duplicate renewal's watchdog

    def connect(self, manifest: dict[str, Any]) -> None:
        discovery = self.transport.request("GET", "/client-interface/v1/discovery")
        fields = closed(discovery, {"protocol_versions", "envelope_schema_versions", "promotion_max_age_ms", "max_media_bytes"})
        if (not isinstance(fields["protocol_versions"], list) or not all(isinstance(v, str) for v in fields["protocol_versions"])
                or not isinstance(fields["envelope_schema_versions"], list) or not all(type(v) is int for v in fields["envelope_schema_versions"])):
            raise ProtocolError("INVALID_ARGUMENT", "invalid_discovery", "Invalid V1 discovery metadata.")
        if "1.0" not in fields["protocol_versions"] or 1 not in fields["envelope_schema_versions"]:
            raise ProtocolError("UNSUPPORTED", "protocol_version", "V1 is not supported by this endpoint.")
        self.promotion_max_age_s = integer(fields["promotion_max_age_ms"], 1) / 1000
        self.max_media_bytes = integer(fields["max_media_bytes"], 1, MAX_MEDIA_BYTES)
        identity = {"client_id": self.credentials.client_id, "embodiment_id": self.credentials.embodiment_id,
                        "application_id": "home-cortex-client", "implementation": {"name": "home-cortex-client",
                        "software_version": "0.1.0", "runtime_version": platform.python_version(), "platform": platform.system()}}
        new_manifest = dict(manifest, revision=1)
        self._control("session.register", {"identity": identity, "manifest": new_manifest})
        with self._lock:
            self._manifest = new_manifest
            self.cursor = None
            self._polled_cursor = None
            self._event_sequences = {}

    def heartbeat(self) -> None:
        self._require_active()
        self._control("session.heartbeat", {})

    def update_capabilities(self, manifest: dict[str, Any]) -> None:
        self._require_active()
        with self._lock:
            if self._manifest is None:
                raise ProtocolError("OFFLINE", "session_offline", "V1 session is unavailable.")
            if manifest["capabilities"] == self._manifest["capabilities"]:
                return
            updated = dict(manifest, revision=self._manifest["revision"] + 1)
        result = self._control("session.capabilities", {"manifest": updated})
        if result["manifest_revision"] != updated["revision"]:
            raise ProtocolError("CONFLICT", "manifest_revision", "V1 manifest revision mismatch.")
        self._manifest = updated

    def _require_active(self) -> None:
        if not self.active:
            raise ProtocolError("OFFLINE", "session_offline", "V1 lease is not active.", retryable=True)

    def poll_commands(self) -> list[dict[str, Any]]:
        self._require_active()
        bound = self.session_id
        query: dict[str, str] = {"session_id": str(bound)}
        if self.cursor is not None:
            query["after"] = self.cursor
        result = closed(self.transport.request("GET", "/client-interface/v1/commands?" + urlencode(query)), {"commands", "next_cursor"})
        if not isinstance(result["commands"], list) or len(result["commands"]) > 128 or not isinstance(result["next_cursor"], str):
            raise ProtocolError("INVALID_ARGUMENT", "invalid_commands", "Invalid V1 command poll.")
        if not self.active or self.session_id != bound:
            raise ProtocolError("CONFLICT", "stale_session", "The poll session has changed.")
        # Advance only after all correlated responses have been acknowledged.
        # A lost POST must leave the batch eligible for non-destructive redelivery.
        self._polled_cursor = result["next_cursor"]
        return result["commands"]

    def fulfill_pending(self, packager: ObservationPackager) -> int:
        count = 0
        bound = self.session_id
        for raw in self.poll_commands():
            # No capture if envelope validation fails. Correlatable failures can be returned safely.
            try:
                command = validate_request(raw)
            except ProtocolError as error:
                # Only a valid correlation ID/operation and current target are
                # sufficient to send a failure. Never capture malformed input.
                if not isinstance(raw, dict) or raw.get("target") != self._target():
                    raise
                identifier(raw.get("request_id"))
                text(raw.get("operation"))
                command = raw
                reply = response(command, self.clock(), error=error)
            else:
                reply = None
            if command["target"] != self._target() or not self.active:
                raise ProtocolError("CONFLICT", "stale_session", "Command has a stale session fence.")
            if reply is None:
                reply = self._execute(command, packager)
            if not self.active or command["target"] != self._target():
                continue  # durable outcome remains; never send under a substituted session
            ack = self.transport.request("POST", MESSAGES, reply, timeout=min(5, max(0.1, self._lease_until - self.monotonic())))
            if ack is not None:
                raise ProtocolError("INVALID_ARGUMENT", "invalid_ack", "V1 command response was not acknowledged.")
            count += 1
        with self._lock:
            if self.session_id == bound and self.active:
                self.cursor = self._polled_cursor
        return count

    def _execute(self, command: dict[str, Any], packager: ObservationPackager) -> dict[str, Any]:
        try:
            first, stored = self.journal.begin(command, self.clock())
            if stored is not None:
                return stored
            if not first:
                raise ProtocolError("INTERNAL_ERROR", "pending_outcome", "Original observation outcome is uncertain; capture was not repeated.")
            if command["operation"] not in {"vision.observe", "vision.observe_clip"}:
                raise ProtocolError("UNSUPPORTED", "unsupported_capability", "The client does not implement this operation.")
            if command["operation"] not in self.effective_capabilities:
                raise ProtocolError("TEMPORARILY_UNAVAILABLE", "capability_unavailable", "The observation capability is unavailable.", retryable=True)
            arguments = command["arguments"]
            if command["operation"] == "vision.observe":
                closed(arguments, set())
                packaged = packager.get_v1_still(timestamp(command["sent_at"]))
            else:
                closed(arguments, {"duration_ms"})
                duration = integer(arguments["duration_ms"], 1, 60000)
                packaged = packager.get_v1_clip(duration)
                end = timestamp(packaged.manifest["captured_end"])
                if abs((end - timestamp(command["sent_at"])).total_seconds()) > 5:
                    raise local_error("evidence_stale")
            if len(packaged.payload) > self.max_media_bytes:
                raise ProtocolError("INVALID_ARGUMENT", "media_too_large", "Evidence exceeds the protocol limit.")
            if self.clock() >= timestamp(command["deadline_at"]):
                raise ProtocolError("TIMEOUT", "deadline", "Observation did not finish before its deadline.")
            reply = response(command, self.clock(), result={"evidence": packaged.manifest,
                             "media_base64": base64.b64encode(packaged.payload).decode("ascii")})
        except EvidenceFailure as error:
            reply = response(command, self.clock(), error=local_error(error.code))
        except ProtocolError as error:
            if error.code == "CONFLICT":
                return response(command, self.clock(), error=error)  # must not overwrite the original receipt
            reply = response(command, self.clock(), error=error)
        except sqlite3.Error:
            raise ProtocolError("INTERNAL_ERROR", "journal_unavailable", "Observation receipt storage is unavailable.") from None
        except Exception:  # noqa: BLE001 — device SDK errors must be sanitized at the wire boundary
            # Device SDK failures must never transmit exception contents.
            reply = response(command, self.clock(), error=local_error("capture_failure"))
        try:
            self.journal.finish(command["request_id"], reply)
        except sqlite3.Error:
            raise ProtocolError("INTERNAL_ERROR", "journal_unavailable", "Observation receipt storage is unavailable.") from None
        return reply

    def publish_evidence(self, value: dict[str, Any]) -> dict[str, Any]:
        self._require_active()
        if "vision.autonomous_promotion" not in self.effective_capabilities:
            raise ProtocolError("TEMPORARILY_UNAVAILABLE", "capability_unavailable", "Promotion is unavailable.", retryable=True)
        # Retry keeps event identity; reconnect gets a new event but preserves the embedded identity/key.
        bound = str(self.session_id)
        key = f'{bound}:{value["evidence"]["evidence_id"]}'
        def make() -> dict[str, Any]:
            sequence = self._event_sequences.get("vision.evidence.publish", 0) + 1
            self._event_sequences["vision.evidence.publish"] = sequence
            base = common("event", self.clock())
            embedded = dict(value, idempotency_key=value.get("idempotency_key", value["evidence"]["evidence_id"]))
            return dict(base, event_id=base["message_id"], source={"client_id": self.credentials.client_id,
                        "embodiment_id": self.credentials.embodiment_id, "session_id": bound}, sequence=sequence,
                        event_type="vision.evidence.publish", occurred_at=value["evidence"]["captured_end"], value=embedded)
        event = self.journal.event(key, make, self.clock())
        embedded = dict(value, idempotency_key=value.get("idempotency_key", value["evidence"]["evidence_id"]))
        if canonical(event["value"]) != canonical(embedded):
            raise ProtocolError("CONFLICT", "idempotency_conflict", "Promotion evidence identity was reused with different content.")
        result = validate_response(self.transport.request("POST", MESSAGES, event, timeout=5), event)
        if result.get("event_id") != event["event_id"] or result.get("disposition") not in {"ACCEPTED", "DUPLICATE"}:
            raise ProtocolError("INVALID_ARGUMENT", "invalid_receipt", "Invalid promotion receipt.")
        return result

    def start_maintenance(self, manifest: Callable[[], dict[str, Any]]) -> None:
        def maintain() -> None:
            while not self._stop.is_set():
                try:
                    if self.auth_failed:
                        self._stop.wait(0.2)
                        continue
                    if not self.active:
                        self.connect(manifest())
                    else:
                        self.update_capabilities(manifest())
                        if self.monotonic() >= self._next_heartbeat:
                            self.heartbeat()
                            self.journal.cleanup(self.clock())
                    self.last_error = None
                except ProtocolError as error:
                    self.note_failure(error)
                    self._stop.wait(1)
                except Exception:  # noqa: BLE001 — fail closed with sanitized local diagnostics
                    self.note_failure(ProtocolError("OFFLINE", "local_failure", "V1 maintenance is unavailable.", retryable=True))
                    self._stop.wait(1)
                self._stop.wait(0.2)
        self._worker = threading.Thread(target=maintain, name="v1-session-lease", daemon=True)
        self._worker.start()

    def note_failure(self, error: ProtocolError) -> None:
        self.last_error = f"{error.code}/{error.detail_code}"
        if error.code in {"PERMISSION_DENIED", "UNSUPPORTED", "INVALID_ARGUMENT"}:
            self.auth_failed = True
        if error.code in {"PERMISSION_DENIED", "OFFLINE", "CONFLICT"}:
            with self._lock:
                self.state = "REPLACED" if error.detail_code == "stale_session" else "STALE"
                self._effective = frozenset()

    def disconnect(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=6)
        try:
            if self.active and not self.auth_failed:
                self._control("session.disconnect", {})
        finally:
            self.state = "DISCONNECTED"
            self._effective = frozenset()
            self.journal.cleanup(self.clock())
            self.journal.close()
