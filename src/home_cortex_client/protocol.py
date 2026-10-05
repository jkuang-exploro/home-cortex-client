"""Independent Client Interface V1 wire codec; no backend implementation imports."""
from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

ERROR_CODES = frozenset({
    "UNSUPPORTED", "OFFLINE", "BUSY", "INVALID_ARGUMENT", "PERMISSION_DENIED",
    "TEMPORARILY_UNAVAILABLE", "SAFETY_REJECTED", "TIMEOUT", "CONFLICT",
    "NOT_FOUND", "INTERNAL_ERROR",
})
COMMON = {"protocol_version", "schema_name", "schema_version", "message_id", "sent_at"}
MAX_MEDIA_BYTES = 32 * 1024 * 1024
MAX_HTTP_BYTES = 46 * 1024 * 1024


class ProtocolError(Exception):
    """Only canonical, sanitized diagnostics leave this boundary."""

    def __init__(self, code: str, detail_code: str, message: str, *, retryable: bool = False):
        self.code = code
        self.detail_code = detail_code
        self.retryable = retryable
        self.permanent = not retryable and code not in {"OFFLINE", "BUSY", "TIMEOUT"}
        super().__init__(message)

    def as_mapping(self) -> dict[str, Any]:
        return {"code": self.code, "detail_code": self.detail_code, "message": str(self),
                    "retryable": self.retryable, "retry_after_ms": None}


def invalid(detail: str = "invalid_envelope") -> ProtocolError:
    return ProtocolError("INVALID_ARGUMENT", detail, "The V1 message is invalid.")


def closed(value: Any, required: set[str], optional: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - (optional or set()):
        raise invalid()
    return value


def stamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})", value,
    ) is None:
        raise invalid("invalid_timestamp")
    try:
        return datetime.fromisoformat(value).astimezone(UTC)
    except ValueError as error:
        raise invalid("invalid_timestamp") from error


def identifier(value: Any) -> str:
    try:
        parsed = UUID(value) if isinstance(value, str) else None
    except ValueError as error:
        raise invalid("invalid_id") from error
    if parsed is None or parsed.version != 4 or str(parsed) != value:
        raise invalid("invalid_id")
    return value


def text(value: Any, *, maximum: int = 200) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or any(ord(c) < 32 for c in value):
        raise invalid("invalid_type")
    return value


def integer(value: Any, minimum: int = 0, maximum: int = 9007199254740991) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise invalid("invalid_type")
    return value


def _numbers(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise invalid("invalid_number")
    if type(value) is int and abs(value) > 9007199254740991:
        raise invalid("invalid_number")
    if isinstance(value, dict):
        for item in value.values():
            _numbers(item)
    elif isinstance(value, list):
        for item in value:
            _numbers(item)


def loads(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for key, value in items:
            if key in obj:
                raise invalid("duplicate_key")
            obj[key] = value
        return obj
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (ValueError, UnicodeError) as error:
        raise invalid("invalid_json") from error
    if not isinstance(value, dict):
        raise invalid("invalid_type")
    _numbers(value)
    return value


def canonical(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: normalize(val) for key, val in item.items()}
        if isinstance(item, list):
            return [normalize(val) for val in item]
        if isinstance(item, float) and item.is_integer():
            return int(item)
        return item
    _numbers(value)
    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def common(schema: str, now: datetime) -> dict[str, Any]:
    return {"protocol_version": "1.0", "schema_name": f"hc.{schema}", "schema_version": 1,
                "message_id": str(uuid4()), "sent_at": stamp(now)}


def request(operation: str, arguments: dict[str, Any], target: dict[str, Any], now: datetime, timeout: float = 10) -> dict[str, Any]:
    base = common("request", now)
    return dict(base, request_id=base["message_id"], target=target, operation=operation,
                arguments=arguments, deadline_at=stamp(now + timedelta(seconds=timeout)))


def response(command: dict[str, Any], now: datetime, *, result: dict[str, Any] | None = None,
             error: ProtocolError | None = None) -> dict[str, Any]:
    base = dict(common("response", now), request_id=command["request_id"],
                target=command["target"], operation=command["operation"], completed_at=stamp(now))
    if error is not None:
        return dict(base, status="FAILED", error=error.as_mapping())
    return dict(base, status="SUCCEEDED", result=result)


def validate_common(value: dict[str, Any], schema: str) -> None:
    if value["protocol_version"] != "1.0" or value["schema_name"] != f"hc.{schema}":
        raise ProtocolError("UNSUPPORTED", "protocol_version", "Unsupported V1 protocol or envelope.")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ProtocolError("UNSUPPORTED", "schema_version", "Unsupported V1 schema version.")
    identifier(value["message_id"])
    timestamp(value["sent_at"])
    if "extensions" in value:
        extensions = value["extensions"]
        if not isinstance(extensions, dict) or len(extensions) > 32 or any(
            re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+", key) is None for key in extensions
        ):
            raise invalid("invalid_extensions")
    _numbers(value)


def validate_target(value: Any) -> dict[str, Any]:
    target = closed(value, {"embodiment_id", "session_id"})
    if target["embodiment_id"] is not None:
        body = text(target["embodiment_id"])
        if not body.startswith("embodiment:") or len(body) <= len("embodiment:"):
            raise invalid("invalid_target")
    if target["session_id"] is not None:
        text(target["session_id"])
    return target


def validate_request(value: Any) -> dict[str, Any]:
    item = closed(value, COMMON | {"request_id", "target", "operation", "arguments", "deadline_at"}, {"extensions"})
    validate_common(item, "request")
    if identifier(item["request_id"]) != item["message_id"]:
        raise invalid("invalid_id")
    validate_target(item["target"])
    text(item["operation"])
    if not isinstance(item["arguments"], dict):
        raise invalid("invalid_arguments")
    span = (timestamp(item["deadline_at"]) - timestamp(item["sent_at"])).total_seconds()
    if not 0 < span <= 3600:
        raise invalid("invalid_deadline")
    if len(canonical(item).encode()) > 65536:
        raise invalid("message_too_large")
    return item


def parse_error(value: Any) -> ProtocolError:
    item = closed(value, {"code", "detail_code", "message", "retryable", "retry_after_ms"})
    if not isinstance(item["code"], str) or item["code"] not in ERROR_CODES or type(item["retryable"]) is not bool:
        raise invalid("invalid_error")
    if not isinstance(item["message"], str) or len(item["message"]) > 500:
        raise invalid("invalid_error")
    detail = item["detail_code"]
    if detail is not None and (not isinstance(detail, str) or re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*", detail) is None):
        raise invalid("invalid_error")
    if item["retry_after_ms"] is not None:
        integer(item["retry_after_ms"])
    # Do not echo arbitrary server/exception text into logs.
    return ProtocolError(item["code"], detail or "remote_failure", "Home Cortex rejected the V1 operation.", retryable=item["retryable"])


def validate_response(value: Any, original: dict[str, Any]) -> dict[str, Any]:
    item = closed(value, COMMON | {"request_id", "target", "operation", "completed_at", "status"}, {"extensions", "result", "error"})
    validate_common(item, "response")
    identifier(item["request_id"])
    validate_target(item["target"])
    timestamp(item["completed_at"])
    expected_id = original.get("request_id", original.get("event_id"))
    expected_op = original.get("operation", original.get("event_type", original.get("telemetry_type")))
    if item["request_id"] != expected_id or item["operation"] != expected_op:
        raise invalid("response_mismatch")
    if "target" in original and item["target"] != original["target"]:
        raise invalid("response_mismatch")
    if "source" in original and item["target"] != {key: original["source"][key] for key in ("embodiment_id", "session_id")}:
        raise invalid("response_mismatch")
    if item["status"] == "FAILED" and "error" in item and "result" not in item:
        raise parse_error(item["error"])
    if item["status"] != "SUCCEEDED" or "error" in item or not isinstance(item.get("result"), dict):
        raise invalid("invalid_response")
    return item["result"]
