"""Standard-library MJPEG-over-HTTP live preview."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


BOUNDARY = "edgeframe"


@dataclass(frozen=True)
class StreamConfig:
    host: str = "127.0.0.1"
    port: int = 8088
    path: str = "/live.mjpg"
    transport: str = "mjpeg-http"

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}{self.path}"

    @property
    def viewer(self) -> str:
        return f"http://{self.host}:{self.port}/"


class MJPEGStreamServer:
    """Serve the latest JPEG as multipart/x-mixed-replace."""

    def __init__(self, runtime: Any, config: StreamConfig) -> None:
        self.runtime = runtime
        self.config = config
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        runtime = self.runtime
        config = self.config

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                if self.path in {"/", "/index.html"}:
                    body = _viewer_html(config.path).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path in {"/health", "/health.json"}:
                    body = runtime.health_json().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path != config.path:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                )
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while runtime.running:
                        frame = runtime.latest_frame()
                        if frame is None:
                            runtime.wait(0.05)
                            continue
                        payload = (
                            f"--{BOUNDARY}\r\n"
                            "Content-Type: image/jpeg\r\n"
                            f"Content-Length: {len(frame.jpeg)}\r\n"
                            f"X-Captured-At: {frame.captured_at}\r\n"
                            "\r\n"
                        ).encode() + frame.jpeg + b"\r\n"
                        self.wfile.write(payload)
                        self.wfile.flush()
                        runtime.wait(runtime.frame_interval)
                except BrokenPipeError:
                    return

        httpd = ThreadingHTTPServer((config.host, config.port), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self._httpd = httpd
        self._thread = thread

    @property
    def bound_port(self) -> int:
        if self._httpd is None:
            return self.config.port
        return int(self._httpd.server_address[1])

    @property
    def endpoint(self) -> str:
        return f"http://{self.config.host}:{self.bound_port}{self.config.path}"

    @property
    def viewer(self) -> str:
        return f"http://{self.config.host}:{self.bound_port}/"

    def stop(self) -> None:
        httpd = self._httpd
        self._httpd = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2)


def _viewer_html(stream: str) -> str:
    return (
        "<!doctype html><meta charset=utf-8><title>Home Cortex Client</title>"
        "<body style='margin:0;background:#111;color:#eee;font:14px sans-serif'>"
        f"<p style='padding:8px'>Home Cortex Client preview — {stream}</p>"
        f"<img src='{stream}' alt='live' style='max-width:100%'>"
        "</body>"
    )

