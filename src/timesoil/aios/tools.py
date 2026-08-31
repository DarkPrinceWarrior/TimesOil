"""Read-only, request-scoped tools exposed to the AIOS agent workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
import json
import re
from typing import Any

from .agents import AgentRole, ToolDefinition, ToolRegistry
from .contracts import (
    Case,
    ContractError,
    ControlAction,
    ControlTarget,
    WellRole,
    WellStatus,
)
from .schedule import ScheduleCompiler, ScheduleError

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
_MAX_ACTIONS = 512

READ_CONTEXT_STATE = "read_context_state"
READ_CONSTRAINTS = "read_constraints"
READ_EVIDENCE_READINESS = "read_evidence_readiness"
VALIDATE_CANDIDATE_CONTROLS = "validate_candidate_controls"

GROUNDED_ROLE_TOOLS: Mapping[AgentRole, tuple[str, ...]] = {
    AgentRole.COORDINATOR: (
        READ_CONTEXT_STATE,
        READ_CONSTRAINTS,
        READ_EVIDENCE_READINESS,
    ),
    AgentRole.ANALYST: (
        READ_CONTEXT_STATE,
        READ_CONSTRAINTS,
        READ_EVIDENCE_READINESS,
    ),
    AgentRole.PLANNER: (
        READ_CONTEXT_STATE,
        READ_CONSTRAINTS,
        READ_EVIDENCE_READINESS,
        VALIDATE_CANDIDATE_CONTROLS,
    ),
    AgentRole.CRITIC: (
        READ_CONTEXT_STATE,
        READ_CONSTRAINTS,
        READ_EVIDENCE_READINESS,
        VALIDATE_CANDIDATE_CONTROLS,
    ),
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
            ToolDefinition(
                VALIDATE_CANDIDATE_CONTROLS,
                "Validate request candidate controls with typed contracts and ScheduleCompiler; does not run GDM or CHDD.",
                _EMPTY_INPUT_SCHEMA,
                _validate_candidate_controls,
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
        "verified": {
            "track1_certified": False,
            "model_z_trained": False,
        },
    }
    try:
        return _bounded_json(payload)
    except _ToolDataError:
        return {
            "source": "current_request_unverified",
            "reported": {},
            "evidence": [],
            "verified": {
                "track1_certified": False,
                "model_z_trained": False,
            },
            "error": "evidence_section_rejected",
        }


def _validate_candidate_controls(
    _: Mapping[str, Any], context: Mapping[str, Any]
) -> Mapping[str, Any]:
    base: dict[str, Any] = {
        "source": "current_request",
        "valid": False,
        "certified": False,
        "gdm_executed": False,
        "chdd_complete": False,
    }
    try:
        case = _case_from_context(context.get("case"))
        actions = _actions_from_context(context.get("candidate_controls"))
        artifact = ScheduleCompiler().compile(case, actions)
    except _ToolDataError as exc:
        return {**base, "error": str(exc)}
    except (ContractError, ScheduleError, TypeError, ValueError):
        return {**base, "error": "strict_contract_violation"}
    return {
        **base,
        "valid": True,
        "action_count": len(artifact.actions),
        "schedule_sha256": artifact.sha256,
    }


def _case_from_context(raw: Any) -> Case:
    required = {"case_id", "start", "end", "economics_start", "producers", "injectors"}
    allowed = required | {"max_liquid_rate"}
    value = _strict_object(raw, required=required, allowed=allowed, code="case_invalid")
    producers = _well_names(value["producers"])
    injectors = _well_names(value["injectors"])
    max_liquid_rate = value.get("max_liquid_rate", 500.0)
    if isinstance(max_liquid_rate, bool) or not isinstance(max_liquid_rate, (int, float)):
        raise _ToolDataError("case_invalid")
    return Case(
        case_id=_string(value["case_id"]),
        start=_month(value["start"]),
        end=_month(value["end"]),
        economics_start=_month(value["economics_start"]),
        producers=producers,
        injectors=injectors,
        max_liquid_rate=float(max_liquid_rate),
    )


def _actions_from_context(raw: Any) -> tuple[ControlAction, ...]:
    if not isinstance(raw, list) or not raw or len(raw) > _MAX_ACTIONS:
        raise _ToolDataError("candidate_controls_invalid")
    fields = {"month", "well", "role", "status", "target", "value"}
    actions: list[ControlAction] = []
    for item in raw:
        value = _strict_object(
            item, required=fields, allowed=fields, code="candidate_controls_invalid"
        )
        target = value["value"]
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise _ToolDataError("candidate_controls_invalid")
        try:
            actions.append(
                ControlAction(
                    month=_month(value["month"]),
                    well=_string(value["well"]),
                    role=WellRole(value["role"]),
                    status=WellStatus(value["status"]),
                    target=ControlTarget(value["target"]),
                    value=float(target),
                )
            )
        except (TypeError, ValueError) as exc:
            raise _ToolDataError("candidate_controls_invalid") from exc
    return tuple(actions)


def _strict_object(
    raw: Any, *, required: set[str], allowed: set[str], code: str
) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
        raise _ToolDataError(code)
    if set(raw) - allowed or required - set(raw):
        raise _ToolDataError(code)
    return raw


def _well_names(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list) or len(raw) > _MAX_CONTAINER_ITEMS:
        raise _ToolDataError("case_invalid")
    return tuple(_string(value) for value in raw)


def _string(raw: Any) -> str:
    if not isinstance(raw, str) or not raw or len(raw) > 128:
        raise _ToolDataError("strict_contract_violation")
    return raw


def _month(raw: Any) -> date:
    if not isinstance(raw, str) or len(raw) != 10:
        raise _ToolDataError("strict_contract_violation")
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise _ToolDataError("strict_contract_violation") from exc


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
