"""Owner-only local V1 identity and standard TLS/CSR provisioning."""
from __future__ import annotations

import json
import os
import ssl
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from .protocol import (
    MAX_HTTP_BYTES,
    ProtocolError,
    closed,
    loads,
    parse_error,
    text,
    timestamp,
)

DEFAULT_STATE_DIR = Path.home() / "Library" / "Application Support" / "Home Cortex Client"


def private_directory(path: Path) -> Path:
    path = path.expanduser().absolute()
    if path.is_symlink():
        raise ValueError("V1 storage must not be a symbolic link")
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    mode = path.stat()
    if mode.st_uid != os.getuid() or stat.S_IMODE(mode.st_mode) & 0o077:
        raise ValueError("V1 storage must be owned by the current user with mode 0700")
    return path


def private_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError("V1 credential files must be regular files")
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("V1 credential files must be owner-only (mode 0600)")
    return path


def write_private(path: Path, payload: bytes) -> None:
    temp = path.with_name(path.name + ".new")
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def https_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
        raise ValueError("V1 endpoint must be an HTTPS origin without credentials, query or path")
    return value.rstrip("/")


@dataclass(frozen=True)
class Credentials:
    client_id: str
    embodiment_id: str
    server_endpoint: str
    root: Path
    expires_at: str
    grants: tuple[dict[str, Any], ...]

    @classmethod
    def load(cls, root: Path, embodiment_id: str) -> Credentials:
        root = private_directory(root)
        meta = loads(private_file(root / "identity.json").read_bytes())
        closed(meta, {"client_id", "embodiment_id", "server_endpoint", "credential_expires_at", "grants"})
        if not text(meta["client_id"]).startswith("client:") or meta["embodiment_id"] != embodiment_id:
            raise ValueError("V1 credentials do not match the configured persistent embodiment")
        endpoint = https_endpoint(meta["server_endpoint"])
        timestamp(meta["credential_expires_at"])
        if not isinstance(meta["grants"], list):
            raise TypeError("V1 credential grants are invalid")
        for name in ("client.crt", "client.key", "ca.crt"):
            private_file(root / name)
        return cls(meta["client_id"], embodiment_id, endpoint, root,
                   meta["credential_expires_at"], tuple(meta["grants"]))

    def context(self) -> ssl.SSLContext:
        context = tls_context(self.root / "ca.crt")
        context.load_cert_chain(str(self.root / "client.crt"), str(self.root / "client.key"))
        return context


def tls_context(ca_file: Path) -> ssl.SSLContext:
    context = ssl.create_default_context(cafile=str(ca_file))
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    return context


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProtocolError("PERMISSION_DENIED", "redirect_forbidden", "V1 endpoint redirects are forbidden.")


class HTTPTransport:
    """No bearer fallback, caller identity headers, insecure TLS or redirects."""

    def __init__(self, endpoint: str, context: ssl.SSLContext):
        self.endpoint = https_endpoint(endpoint)
        self.opener = build_opener(HTTPSHandler(context=context), NoRedirect())

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None,
                *, timeout: float = 5) -> dict[str, Any] | None:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        if data is not None and len(data) > MAX_HTTP_BYTES:
            raise ProtocolError("INVALID_ARGUMENT", "message_too_large", "V1 media exceeds the wire limit.")
        headers = {} if data is None else {"Content-Type": "application/json"}
        req = Request(self.endpoint + path, data=data, headers=headers, method=method)
        try:
            with self.opener.open(req, timeout=timeout) as reply:
                raw = reply.read(MAX_HTTP_BYTES + 1)
                if len(raw) > MAX_HTTP_BYTES:
                    raise ProtocolError("INVALID_ARGUMENT", "message_too_large", "V1 reply exceeds the wire limit.")
                if reply.status == 204:
                    return None
                return loads(raw)
        except HTTPError as error:
            try:
                body = loads(error.read(65537))
                remote = parse_error(body.get("error"))
            except ProtocolError:
                remote = ProtocolError("PERMISSION_DENIED" if error.code in {401, 403} else "OFFLINE",
                                       "http_failure", "Home Cortex rejected the request.",
                                       retryable=error.code >= 500)
            raise remote from None
        except (URLError, OSError) as error:
            if isinstance(error, ssl.SSLError) or isinstance(getattr(error, "reason", None), ssl.SSLError):
                raise ProtocolError("PERMISSION_DENIED", "tls_validation_failed", "V1 TLS authentication failed.") from None
            raise ProtocolError("OFFLINE", "transport_unavailable", "Home Cortex is unavailable.", retryable=True) from None


def enroll(invitation_path: Path, ca_file: Path, root: Path, *, embodiment_id: str) -> Credentials:
    """Enrollment secrets arrive via a protected file, never CLI arguments/logs."""
    invitation = loads(private_file(invitation_path.expanduser()).read_bytes())
    closed(invitation, {"invitation_id", "token", "bootstrap_endpoint", "embodiment_id"})
    if invitation["embodiment_id"] != embodiment_id:
        raise ValueError("Invitation does not name the existing configured embodiment")
    root = private_directory(root)
    if (root / "identity.json").exists():
        raise ValueError("V1 identity already exists; do not overwrite a provisioned client")
    key = root / "client.key"
    csr = root / "client.csr"
    if not key.exists():
        # Standard OpenSSL algorithms; private material never enters stdout/argv.
        fd = os.open(key, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        _openssl(["genpkey", "-algorithm", "EC", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-out", str(key)])
    private_file(key)
    if not csr.exists():
        fd = os.open(csr, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        _openssl(["req", "-new", "-key", str(key), "-subj", "/CN=home-cortex-client", "-out", str(csr)])
    result = HTTPTransport(invitation["bootstrap_endpoint"], tls_context(ca_file)).request(
        "POST", "/client-interface/v1/enroll", {"invitation_id": invitation["invitation_id"],
            "token": invitation["token"], "csr_pem": private_file(csr).read_text()}, timeout=10,
    )
    bundle = closed(result, {"client_id", "embodiment_id", "certificate_pem", "ca_chain_pem", "server_endpoint", "protocol_versions", "grants", "credential_expires_at"})
    if bundle["embodiment_id"] != embodiment_id or "1.0" not in bundle["protocol_versions"]:
        raise ValueError("Enrollment returned an incompatible identity")
    https_endpoint(bundle["server_endpoint"])
    # Authenticate newly issued certificate/key against the originally trusted CA.
    cert = bundle["certificate_pem"]
    if not isinstance(cert, str) or len(cert) > 32000 or "-----BEGIN CERTIFICATE-----" not in cert:
        raise ValueError("Enrollment certificate is invalid")
    write_private(root / "client.crt", cert.encode())
    write_private(root / "ca.crt", ca_file.read_bytes())
    meta = {k: bundle[k] for k in ("client_id", "embodiment_id", "server_endpoint", "credential_expires_at", "grants")}
    tls_context(root / "ca.crt").load_cert_chain(str(root / "client.crt"), str(key))
    write_private(root / "identity.json", json.dumps(meta, indent=2).encode())
    return Credentials.load(root, embodiment_id)


def _openssl(arguments: list[str]) -> None:
    result = subprocess.run(["openssl", *arguments], capture_output=True, check=False)
    if result.returncode:
        raise ValueError("OpenSSL credential generation failed; no secret output is logged")
