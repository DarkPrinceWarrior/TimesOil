"""Read-only, request-scoped tools exposed to the AIOS agent workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import re
from typing import Any

from .agents import AgentRole, ToolDefinition, ToolRegistry

_EMPTY_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}
_STATE_KEYS = ("case_id", "track", "month", "facts", "field_state", "state")
_SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:api[_-]?key|authorization|cookie|credential|password|secret|token)(?:$|_)",
    re.IGNORECASE,
)
_MAX_SECTION_CHARS = 24_000
_MAX_CONTAINER_ITEMS = 512
_MAX_DEPTH = 8
_MAX_STRING_CHARS = 2_048

READ_CONTEXT_STATE = "read_context_state"
READ_CONSTRAINTS = "read_constraints"
READ_EVIDENCE_READINESS = "read_evidence_readiness"

_READ_ONLY_TOOLS = (READ_CONTEXT_STATE, READ_CONSTRAINTS, READ_EVIDENCE_READINESS)

GROUNDED_ROLE_TOOLS: Mapping[AgentRole, tuple[str, ...]] = {
    role: _READ_ONLY_TOOLS for role in AgentRole
}


class _ToolDataError(ValueError):
    pass


def build_grounded_tool_registry() -> ToolRegistry:
    """Create stateless tools; request context is supplied only at execution time."""

    return ToolRegistry(
        (
            ToolDefinition(
                READ_CONTEXT_STATE,
                "Read normalized field state and facts from this request only.",
                _EMPTY_INPUT_SCHEMA,
                _read_context_state,
            ),
            ToolDefinition(
                READ_CONSTRAINTS,
                "Read control constraints from this request only.",
                _EMPTY_INPUT_SCHEMA,
                _read_constraints,
            ),
            ToolDefinition(
                READ_EVIDENCE_READINESS,
                "Read reported evidence and fail-closed readiness from this request only.",
                _EMPTY_INPUT_SCHEMA,
                _read_evidence_readiness,
            ),
        )
    )


def _read_context_state(
    _: Mapping[str, Any], context: Mapping[str, Any]
) -> Mapping[str, Any]:
    selected = {key: context[key] for key in _STATE_KEYS if key in context}
    return _safe_result("state", selected)


def _read_constraints(
    _: Mapping[str, Any], context: Mapping[str, Any]
) -> Mapping[str, Any]:
    if "constraints" in context:
        return _safe_result("constraints", context["constraints"])
    if "case" in context:
        return _safe_result("constraints", context["case"])
    return {"source": "current_request", "available": False, "constraints": None}


def _read_evidence_readiness(
    _: Mapping[str, Any], context: Mapping[str, Any]
) -> Mapping[str, Any]:
    reported = context.get("readiness", {})
    facts = context.get("facts")
    if not reported and isinstance(facts, Mapping):
        reported = {
            key: value
            for key, value in facts.items()
            if any(marker in key.lower() for marker in ("ready", "certified", "trained"))
        }
    payload = {
        "source": "current_request_unverified",
        "reported": reported,
        "evidence": context.get("evidence", []),
        "verified": {},
    }
    try:
        return _bounded_json(payload)
    except _ToolDataError:
        return {
            "source": "current_request_unverified",
            "reported": {},
            "evidence": [],
            "verified": {},
            "error": "evidence_section_rejected",
        }


def _safe_result(label: str, value: Any) -> Mapping[str, Any]:
    try:
        safe = _bounded_json(value)
    except _ToolDataError:
        return {
            "source": "current_request",
            "available": False,
            label: None,
            "error": f"{label}_section_rejected",
        }
    return {"source": "current_request", "available": bool(value), label: safe}


def _bounded_json(value: Any) -> Any:
    safe = _sanitize(value, depth=0)
    encoded = json.dumps(safe, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if len(encoded) > _MAX_SECTION_CHARS:
        raise _ToolDataError("section_too_large")
    return json.loads(encoded)


def _sanitize(value: Any, *, depth: int) -> Any:
    if depth > _MAX_DEPTH:
        raise _ToolDataError("section_too_deep")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > _MAX_STRING_CHARS:
            raise _ToolDataError("string_too_large")
        return value
    if isinstance(value, Mapping):
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise _ToolDataError("object_too_large")
        return {
            key: _sanitize(item, depth=depth + 1)
            for key, item in value.items()
            if isinstance(key, str)
            and len(key) <= 128
            and not _SENSITIVE_KEY.search(key)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise _ToolDataError("array_too_large")
        return [_sanitize(item, depth=depth + 1) for item in value]
    raise _ToolDataError("unsupported_json_value")
