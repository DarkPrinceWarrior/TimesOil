from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient

from timesoil.aios.agents import (
    AgentWorkflow,
    ToolNotAllowedError,
    WorkflowError,
)
from timesoil.aios.api import app, get_agent_workflow
from timesoil.aios.llm import ChatMessage, LLMResponse, ToolCall
from timesoil.aios.tools import (
    GROUNDED_ROLE_TOOLS,
    READ_CONTEXT_STATE,
    READ_EVIDENCE_READINESS,
    VALIDATE_CANDIDATE_CONTROLS,
    build_grounded_tool_registry,
)


def _context() -> dict[str, Any]:
    return {
        "case_id": "model-z",
        "track": 2,
        "month": "2014-01-01",
        "facts": {"field_oil_rate": 100.0},
        "constraints": {"fixed_total_injection": True},
        "readiness": {
            "model_z_trained": True,
        },
        "evidence": ["request assertion only"],
        "case": {
            "case_id": "model-z",
            "start": "2014-01-01",
            "end": "2014-02-01",
            "economics_start": "2014-01-01",
            "producers": ["P-1"],
            "injectors": ["I-1"],
            "max_liquid_rate": 500.0,
        },
        "candidate_controls": [
            {
                "month": "2014-01-01",
                "well": "P-1",
                "role": "producer",
                "status": "OPEN",
                "target": "LRAT",
                "value": 120.0,
            },
            {
                "month": "2014-01-01",
                "well": "I-1",
                "role": "injector",
                "status": "OPEN",
                "target": "WRAT",
                "value": 80.0,
            },
        ],
    }


def test_grounded_tools_are_request_scoped_redacted_and_fail_closed() -> None:
    registry = build_grounded_tool_registry()
    context = _context()
    context["facts"]["api_key"] = "must-not-leak"
    allowed = tuple(registry.names)

    state = asyncio.run(
        registry.execute(
            ToolCall("state-1", READ_CONTEXT_STATE, {}),
            allowed=allowed,
            context=context,
        )
    ).output
    readiness = asyncio.run(
        registry.execute(
            ToolCall("ready-1", READ_EVIDENCE_READINESS, {}),
            allowed=allowed,
            context=context,
        )
    ).output
    validation = asyncio.run(
        registry.execute(
            ToolCall("validate-1", VALIDATE_CANDIDATE_CONTROLS, {}),
            allowed=allowed,
            context=context,
        )
    ).output

    assert state["state"]["facts"] == {"field_oil_rate": 100.0}
    assert readiness["reported"]["model_z_trained"] is True
    assert readiness["verified"] == {"model_z_trained": False}
    assert validation["valid"] is True
    assert validation["certified"] is False
    assert validation["simulator_executed"] is False
    assert validation["chdd_complete"] is False
    assert len(validation["schedule_sha256"]) == 64


def test_tool_allowlist_schema_and_payload_limits_fail_closed() -> None:
    registry = build_grounded_tool_registry()
    allowed = (READ_CONTEXT_STATE,)

    with pytest.raises(ToolNotAllowedError):
        asyncio.run(
            registry.execute(
                ToolCall("bad-1", "shell", {}), allowed=allowed, context={}
            )
        )
    with pytest.raises(WorkflowError, match="strict schema"):
        asyncio.run(
            registry.execute(
                ToolCall("bad-2", READ_CONTEXT_STATE, {"extra": True}),
                allowed=allowed,
                context={},
            )
        )
    with pytest.raises(WorkflowError, match="bounded contract"):
        asyncio.run(
            registry.execute(
                ToolCall("bad-3", READ_CONTEXT_STATE, {"value": "x" * 9_000}),
                allowed=allowed,
                context={},
            )
        )
    with pytest.raises(WorkflowError, match="call id"):
        asyncio.run(
            registry.execute(
                ToolCall("x" * 129, READ_CONTEXT_STATE, {}),
                allowed=allowed,
                context={},
            )
        )


class _ToolCallingLLM:
    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        reasoning: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMResponse:
        del reasoning, tool_choice, max_tokens, timeout_seconds
        assert tools
        available = {item["function"]["name"] for item in tools}
        name = (
            VALIDATE_CANDIDATE_CONTROLS
            if "планировщик" in messages[0].content
            else READ_CONTEXT_STATE
        )
        assert name in available
        call = ToolCall(f"{name}-1", name, {})
        return LLMResponse("grounded", "private reasoning", "tool_calls", (call,))

    async def structured(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: Mapping[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[dict[str, Any], LLMResponse]:
        del messages, schema_name, max_tokens, timeout_seconds
        role = schema["properties"]["role"]["const"]
        payload = {
            "role": role,
            "summary": "grounded",
            "recommendation": "hold",
            "evidence": ["tool trace"],
            "approved": False,
        }
        return payload, LLMResponse("{}", None, "stop")


def test_agent_api_executes_grounded_tool_and_returns_safe_trace() -> None:
    workflow = AgentWorkflow(
        _ToolCallingLLM(),
        build_grounded_tool_registry(),
        role_tools=GROUNDED_ROLE_TOOLS,
    )
    app.dependency_overrides[get_agent_workflow] = lambda: workflow
    try:
        response = TestClient(app).post(
            "/v1/experiments/agents", json={"context": _context()}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    planner = next(item for item in payload["decisions"] if item["role"] == "planner")
    assert planner["tools"][0]["tool"] == VALIDATE_CANDIDATE_CONTROLS
    assert planner["tools"][0]["output"]["valid"] is True
    assert "private reasoning" not in response.text
    assert "must-not-leak" not in response.text
