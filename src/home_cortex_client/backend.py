"""Session-only Home Cortex transport for an already configured embodiment."""
from __future__ import annotations

import base64
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

    def connect(self, available_capabilities: list[str] | None = None) -> None:
        names = ["vision.observe"] if available_capabilities is None else list(available_capabilities)
        response = self._request(
            "POST", "", {"available_capabilities": names},
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

    def update_capabilities(self, available_capabilities: list[str]) -> None:
        if self.session_id is None:
            raise RuntimeError("Home Cortex session is not open")
        self._request(
            "POST", "/capabilities",
            {"available_capabilities": list(available_capabilities)},
        )

    def poll_commands(self) -> list[dict]:
        if self.session_id is None:
            raise RuntimeError("Home Cortex session is not open")
        result = self._request("GET", "/commands")
        commands = result.get("commands")
        if not isinstance(commands, list):
            raise RuntimeError("Home Cortex returned an invalid command list")
        return [command for command in commands if isinstance(command, dict)]

    def submit_observation(self, command_id: str, manifest: dict, media: bytes) -> dict:
        return self._request(
            "POST",
            "/observations",
            {
                "command_id": command_id,
                "manifest": manifest,
                "media_base64": base64.b64encode(media).decode("ascii"),
            },
            timeout=60,
        )

    def submit_observation_failure(self, command_id: str, code: str, message: str) -> dict:
        return self._request(
            "POST",
            "/observations",
            {"command_id": command_id, "error": {"code": code, "message": message}},
        )

    def disconnect(self) -> None:
        if self.session_id is None:
            return
        try:
            self._request("DELETE", "")
        finally:
            self.session_id = None

    def _request(
        self, method: str, suffix: str, payload: dict | None = None, *, timeout: float = 5,
    ) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.session_id is not None:
            headers["X-Embodiment-Session-ID"] = self.session_id
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.url + suffix, data=body, headers=headers, method=method)
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise RuntimeError("Home Cortex returned an invalid session response")
        return result
