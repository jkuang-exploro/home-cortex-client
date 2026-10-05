"""Independent V1 codec, lease, evidence and durable replay checks."""
from __future__ import annotations

import base64
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from home_cortex_client.buffer import RingBuffer
from home_cortex_client.credentials import (
    Credentials,
    https_endpoint,
    private_directory,
)
from home_cortex_client.evidence import (
    EvidenceFailure,
    EvidencePackager,
    LocalEvidenceStore,
    PackagedEvidence,
    build_manifest,
    decode_clip,
    encode_clip,
)
from home_cortex_client.protocol import (
    ProtocolError,
    loads,
    request,
    response,
    stamp,
    validate_request,
    validate_response,
)
from home_cortex_client.receipts import ReceiptJournal
from home_cortex_client.v1 import V1Session, capability_manifest

NOW = datetime(2026, 10, 5, 4, 0, 10, tzinfo=UTC)
BODY = "embodiment:macbook-0"
TARGET = {"embodiment_id": BODY, "session_id": "runtime-session:test:1"}
PROFILE = {"revision": 1, "capabilities": [{"name": name, "schema_version": 1, "availability": "AVAILABLE"} for name in ("vision.observe", "vision.observe_clip", "vision.autonomous_promotion")]}
VECTORS = Path(__file__).parent / "vectors" / "v1"


class Peer:
    def __init__(self):
        self.now = NOW
        self.monotonic = 100.0
        self.calls = []
        self.commands = []
        self.replies = []
        self.generation = 0
        self.revision = 1
        self.expiry = NOW + timedelta(seconds=60)
        self.lose_ack = False
        self.effective = []

    def request(self, method, path, payload=None, *, timeout=5):
        self.calls.append((method, path, deepcopy(payload)))
        if path.endswith("discovery"):
            return {"protocol_versions": ["1.0"], "envelope_schema_versions": [1], "promotion_max_age_ms": 21600000, "max_media_bytes": 33554432}
        if method == "GET":
            return {"commands": self.commands, "next_cursor": "opaque-cursor"}
        assert payload is not None
        if payload["schema_name"] == "hc.response":
            self.replies.append(payload)
            if self.lose_ack:
                self.lose_ack = False
                raise ProtocolError("OFFLINE", "transport_unavailable", "Reply acknowledgement was lost.")
            return None
        if payload["schema_name"] == "hc.event":
            return dict(response({"request_id": payload["event_id"], "target": payload["source"], "operation": payload["event_type"]}, self.now,
                                 result={"event_id": payload["event_id"], "disposition": "ACCEPTED", "received_at": stamp(self.now)}),
                        target={"embodiment_id": BODY, "session_id": self.session})
        if payload["operation"] == "session.register":
            self.generation += 1
        if payload["operation"] in {"session.register", "session.capabilities"}:
            self.effective = [cap["name"] for cap in payload["arguments"]["manifest"]["capabilities"] if cap["availability"] == "AVAILABLE"]
        if payload["operation"] == "session.capabilities":
            self.revision = payload["arguments"]["manifest"]["revision"]
        state = "DISCONNECTED" if payload["operation"] == "session.disconnect" else "ACTIVE"
        self.expiry = self.now + timedelta(seconds=60)
        view = {"client_id": "client:test", "embodiment_id": BODY, "session_id": self.session, "state": state,
                    "connected_at": stamp(self.now), "last_seen_at": stamp(self.now), "server_time": stamp(self.now),
                    "lease_expires_at": stamp(self.expiry), "heartbeat_interval_ms": 20000, "lease_duration_ms": 60000,
                    "manifest_revision": self.revision, "effective_capabilities": self.effective}
        return response(payload, self.now, result=view)

    @property
    def session(self):
        return f"runtime-session:test:{self.generation}"

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)
        self.monotonic += seconds


def session(tmp_path):
    peer = Peer()
    credentials = Credentials("client:test", BODY, "https://localhost:8443", tmp_path,
                              stamp(NOW + timedelta(days=30)), ())
    client = V1Session(credentials, tmp_path / "state", transport=peer, clock=lambda: peer.now,
                       monotonic=lambda: peer.monotonic)
    client.connect(PROFILE)
    return client, peer


class Packager:
    def __init__(self):
        self.count = 0
    def get_v1_still(self, sent_at):
        self.count += 1
        return PackagedEvidence(manifest={"captured_end": stamp(sent_at)}, payload=b"jpeg")

    def get_v1_clip(self, duration_ms):
        return self.get_v1_still(NOW)


def test_codec_rejects_malformed_commands_and_preserves_extensions():
    good = request("vision.observe", {}, TARGET, NOW)
    good["extensions"] = {"test.note": "advisory"}
    assert validate_request(good) == good
    for change in ({"unknown": True}, {"request_id": "bad"}, {"schema_version": True},
                   {"deadline_at": stamp(NOW)}, {"arguments": []}, {"extensions": {"unnamespaced": True}}):
        with pytest.raises(ProtocolError):
            validate_request(dict(good, **change))
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'{"a":9007199254740992}'):
        with pytest.raises(ProtocolError):
            loads(raw)
    with pytest.raises(ProtocolError):
        validate_response(response(good, NOW, result={}), dict(good, operation="audio.speak"))


def test_server_driven_lease_and_reconnect_fence(tmp_path):
    client, peer = session(tmp_path)
    assert client.active and client._next_heartbeat == 120
    peer.advance(58)
    assert not client.active and client.effective_capabilities == frozenset()
    with pytest.raises(ProtocolError, match="not active"):
        client.poll_commands()
    client.connect({"revision": 1, "capabilities": []})
    assert client.active and client.session_id != TARGET["session_id"]
    peer.commands = [request("vision.observe", {}, TARGET, peer.now)]
    packager = Packager()
    with pytest.raises(ProtocolError) as failure:
        client.fulfill_pending(packager)
    assert failure.value.detail_code == "stale_session" and packager.count == 0
    client.disconnect()


def test_duplicate_observation_survives_redelivery_transport_and_process_restart(tmp_path):
    client, peer = session(tmp_path)
    command = request("vision.observe", {}, TARGET, NOW)
    peer.commands = [command]
    packager = Packager()
    client.fulfill_pending(packager)
    client.fulfill_pending(packager)
    assert packager.count == 1 and peer.replies[0] == peer.replies[1]
    assert "after=opaque-cursor" in peer.calls[-2][1]
    client.journal.close()
    recovered = ReceiptJournal(tmp_path / "state")
    first, stored = recovered.begin(command, NOW)
    assert not first and stored == peer.replies[0]
    with pytest.raises(ProtocolError) as conflict:
        recovered.begin(dict(command, operation="vision.observe_clip"), NOW)
    assert conflict.value.code == "CONFLICT"
    recovered.close()


def test_interrupted_pending_receipt_does_not_repeat_capture(tmp_path):
    client, _peer = session(tmp_path)
    command = request("vision.observe", {}, TARGET, NOW)
    client.journal.begin(command, NOW)
    packager = Packager()
    reply = client._execute(command, packager)
    assert reply["error"]["detail_code"] == "pending_outcome" and packager.count == 0
    client.disconnect()


def test_lost_ack_redelivers_without_recapture_and_malformed_command_returns_error(tmp_path):
    client, peer = session(tmp_path)
    command = request("vision.observe", {}, TARGET, NOW)
    peer.commands = [command]
    peer.lose_ack = True
    packager = Packager()
    with pytest.raises(ProtocolError):
        client.fulfill_pending(packager)
    assert client.cursor is None
    client.fulfill_pending(packager)
    assert packager.count == 1 and peer.replies[0] == peer.replies[1]
    peer.commands = [dict(request("vision.observe", {}, TARGET, NOW), unknown_authority=True)]
    client.fulfill_pending(packager)
    assert peer.replies[-1]["error"]["code"] == "INVALID_ARGUMENT" and packager.count == 1
    client.disconnect()


def test_receipt_storage_rejects_competing_process_and_identity_reuse(tmp_path):
    journal = ReceiptJournal(tmp_path / "state", identity="client:a")
    with pytest.raises(ProtocolError) as busy:
        ReceiptJournal(tmp_path / "state", identity="client:a")
    assert busy.value.code == "BUSY"
    journal.close()
    with pytest.raises(ProtocolError) as denied:
        ReceiptJournal(tmp_path / "state", identity="client:b")
    assert denied.value.detail_code == "state_identity_mismatch"


def test_capture_finishing_after_replacement_does_not_send_or_advance_new_cursor(tmp_path):
    client, peer = session(tmp_path)
    peer.commands = [request("vision.observe", {}, TARGET, NOW)]
    class ReplacedCapture(Packager):
        def get_v1_still(self, sent_at):
            client.connect(PROFILE)
            return super().get_v1_still(sent_at)
    packager = ReplacedCapture()
    client.fulfill_pending(packager)
    assert client.session_id != TARGET["session_id"]
    assert client.cursor is None and peer.replies == [] and packager.count == 1
    client.disconnect()


def test_deadlines_unsupported_and_sanitized_device_failure(tmp_path):
    client, _peer = session(tmp_path)
    old = request("vision.observe", {}, TARGET, NOW - timedelta(seconds=30))
    packager = Packager()
    assert client._execute(old, packager)["error"]["code"] == "TIMEOUT"
    unsupported = request("mobility.goto", {}, TARGET, NOW)
    assert client._execute(unsupported, packager)["error"]["code"] == "UNSUPPORTED"
    def broken(sent_at):
        raise RuntimeError("secret must never cross the protocol")
    packager.get_v1_still = broken
    reply = client._execute(request("vision.observe", {}, TARGET, NOW), packager)
    assert reply["error"]["code"] == "INTERNAL_ERROR" and "secret" not in json.dumps(reply)
    assert packager.count == 0
    client.disconnect()


def test_manifest_support_and_revisions(tmp_path):
    runtime = SimpleNamespace(camera_status=lambda: {"buffer_duration": 0, "failure_code": "camera_permission_denied"},
                              local_capabilities=lambda: {"vision.observe": False, "vision.observe_clip": False},
                              buffer=SimpleNamespace(duration_s=60))
    manifest = capability_manifest(runtime, ("vision.observe", "vision.observe_clip", "vision.autonomous_promotion"))
    assert len(manifest["capabilities"]) == 3
    assert all(item["availability"] == "TEMPORARILY_UNAVAILABLE" for item in manifest["capabilities"])
    assert manifest["capabilities"][0]["reason"]["code"] == "PERMISSION_DENIED"
    runtime.buffer = SimpleNamespace(duration_s=60)
    client, peer = session(tmp_path)
    client.update_capabilities(manifest)
    assert client._manifest is not None
    assert client._manifest["revision"] == 2
    count = len(peer.calls)
    client.update_capabilities(manifest)
    assert len(peer.calls) == count
    client.disconnect()


def test_promotion_retries_have_identical_event_and_embedded_identity(tmp_path):
    client, peer = session(tmp_path)
    value = {"evidence": {"evidence_id": "evidence:test", "captured_end": stamp(NOW)}, "promotion": {"decision": "PROMOTE"}}
    client.publish_evidence(value)
    client.publish_evidence(value)
    events = [call[2] for call in peer.calls if call[2] and call[2]["schema_name"] == "hc.event"]
    assert events[0] == events[1]
    assert events[0]["value"]["evidence"] == value["evidence"]
    client.note_failure(ProtocolError("PERMISSION_DENIED", "certificate_revoked", "Revoked"))
    assert not client.active and client.auth_failed
    client.disconnect()


@pytest.mark.parametrize("filename", ["visual-evidence.json", "visual-evidence-unicode.json", "hcclip1.json"])
def test_backend_golden_evidence_and_hcclip_bytes(filename):
    vector = json.loads((VECTORS / filename).read_text())
    if filename == "visual-evidence-unicode.json":
        fields = vector["manifest_without_evidence_id"]
        assert build_manifest(**fields)["evidence_id"] == vector["evidence_id"]
        assert json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False) == vector["canonical_manifest_utf8"]
        return
    # Golden documents contain the manifest plus media bytes; use unchanged identity algorithm.
    manifest = vector.get("evidence", vector.get("manifest"))
    media = base64.b64decode(vector.get("media_base64", vector.get("container_base64")))
    fields = {key: value for key, value in manifest.items() if key != "evidence_id"}
    assert build_manifest(**fields) == manifest
    if filename == "hcclip1.json":
        from home_cortex_client.buffer import CapturedSegment
        frames = decode_clip(media)
        segments = tuple(CapturedSegment(manifest["camera_id"], seq, datetime.fromtimestamp(millis / 1000, UTC),
                         1, manifest["width"], manifest["height"], "jpeg", "image/jpeg", "test", jpeg)
                         for seq, millis, jpeg in frames)
        assert encode_clip(segments) == media


def test_v1_clip_exact_span_and_fresh_still(tmp_path):
    buffer = RingBuffer()
    for i in range(5):
        buffer.append_frame(camera_id="camera:test", captured_at=NOW + timedelta(seconds=i),
                            width=1, height=1, payload=b"\xff\xd8\xff\xd9", duration_s=1)
    packager = EvidencePackager(buffer, LocalEvidenceStore(tmp_path), embodiment_id=BODY,
                               clock=lambda: NOW + timedelta(seconds=4))
    selected = packager.get_v1_clip(3000)
    assert selected.manifest["duration_ms"] == 3000
    assert decode_clip(selected.payload)[-1][1] == int((NOW + timedelta(seconds=4)).timestamp() * 1000)
    with pytest.raises(EvidenceFailure) as failure:
        packager.get_v1_clip(2500)
    assert failure.value.code == "buffer_too_short"
    with pytest.raises(EvidenceFailure):
        packager.get_v1_still(NOW + timedelta(seconds=5))


def test_https_only_and_protected_storage(tmp_path):
    for endpoint in ("http://localhost", "https://key@localhost", "https://localhost/path", "https://localhost?token=x"):
        with pytest.raises(ValueError):
            https_endpoint(endpoint)
    tmp_path.chmod(0o755)
    with pytest.raises(ValueError):
        private_directory(tmp_path)
