"""Session-only Home Cortex transport for an already configured embodiment."""
from __future__ import annotations

import json
from urllib.parse import quote
from urllib.request import Request, urlopen


class BackendSession:
    """Identify a body to Home Cortex; never create or assign it."""

    def __init__(self, base_url: str, api_key: str, embodiment_id: str) -> None:
        if not base_url or not api_key:
            raise ValueError("Home Cortex URL and API key are both required")
        if not embodiment_id.startswith("embodiment:"):
            raise ValueError("embodiment_id must be an embodiment: record ID")
        self.url = (
            base_url.rstrip("/") + "/v1/embodiments/"
            + quote(embodiment_id, safe=":") + "/session"
        )
        self.api_key = api_key
        self.embodiment_id = embodiment_id
        self.session_id: str | None = None

    def connect(self) -> None:
        response = self._request(
            "POST", "", {"available_capabilities": ["vision.observe"]},
        )
        session = response.get("session")
        session_id = session.get("session_id") if isinstance(session, dict) else None
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("Home Cortex did not return a runtime session ID")
        self.session_id = session_id

    def heartbeat(self) -> None:
        if self.session_id is None:
            raise RuntimeError("Home Cortex session is not open")
        self._request("POST", "/heartbeat")

    def disconnect(self) -> None:
        if self.session_id is None:
            return
        try:
            self._request("DELETE", "")
        finally:
            self.session_id = None

    def _request(self, method: str, suffix: str, payload: dict | None = None) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.session_id is not None:
            headers["X-Embodiment-Session-ID"] = self.session_id
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.url + suffix, data=body, headers=headers, method=method)
        with urlopen(request, timeout=5) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise RuntimeError("Home Cortex returned an invalid session response")
        return result
