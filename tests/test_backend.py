"""A camera client can claim only a pre-existing embodiment runtime session."""
from io import BytesIO
import json
from urllib.error import HTTPError

import pytest

from home_cortex_client.backend import BackendSession


def test_session_uses_stable_id_and_never_writes_identity(monkeypatch) -> None:
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        assert timeout == 5
        if request.method == "POST" and request.full_url.endswith("/session"):
            return BytesIO(json.dumps({"session": {"session_id": "runtime-session:1"}}).encode())
        return BytesIO(b"{}")

    monkeypatch.setattr("home_cortex_client.backend.urlopen", fake_urlopen)
    session = BackendSession("http://cortex.local/", "test-key", "embodiment:macbook-0")
    session.connect()
    session.heartbeat()
    session.disconnect()
    assert session.session_id is None
    assert [(request.method, request.full_url) for request in calls] == [
        ("POST", "http://cortex.local/v1/embodiments/embodiment:macbook-0/session"),
        ("POST", "http://cortex.local/v1/embodiments/embodiment:macbook-0/session/heartbeat"),
        ("DELETE", "http://cortex.local/v1/embodiments/embodiment:macbook-0/session"),
    ]
    assert json.loads(calls[0].data) == {"available_capabilities": ["vision.observe"]}
    assert calls[1].get_header("X-embodiment-session-id") == "runtime-session:1"
    assert calls[2].get_header("X-embodiment-session-id") == "runtime-session:1"
    assert all(request.get_header("Authorization") == "Bearer test-key" for request in calls)


def test_unknown_body_cannot_be_created_by_client(monkeypatch) -> None:
    def unknown(request, timeout):
        raise HTTPError(request.full_url, 404, "unknown_embodiment", {}, None)

    monkeypatch.setattr("home_cortex_client.backend.urlopen", unknown)
    session = BackendSession("http://cortex.local", "test-key", "embodiment:missing")
    with pytest.raises(HTTPError) as error:
        session.connect()
    assert error.value.code == 404
    assert session.session_id is None
