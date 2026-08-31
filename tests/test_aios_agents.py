from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import httpx
import pytest

from timesoil.aios.agents import (
    ROLE_ORDER,
    AgentRole,
    AgentWorkflow,
    ToolDefinition,
    ToolNotAllowedError,
    ToolRegistry,
    WorkflowError,
)
from timesoil.aios.economics import CHDD_FIELDS, CHDDEconomicsAdapter
from timesoil.aios.llm import (
    APPROVED_MODEL,
    ChatMessage,
    ExternalQwenClient,
    LLMConfig,
    LLMError,
    LLMResponse,
    ToolCall,
)

_TEST_BASE_URL = "https://qwen.example/v1"


def _http_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=_TEST_BASE_URL + "/",
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )


def _config() -> LLMConfig:
    return LLMConfig(
        api_key="test-only-key", base_url=_TEST_BASE_URL, timeout_seconds=2
    )


def test_llm_config_is_qwen36_only_and_hides_secret() -> None:
    config = _config()
    assert config.model == APPROVED_MODEL
    assert "test-only-key" not in repr(config)
    with pytest.raises(ValueError, match="external HTTPS"):
        LLMConfig(api_key="x", base_url="http://qwen.example/v1")
    with pytest.raises(ValueError, match="qwen3.6"):
        LLMConfig(api_key="x", base_url=_TEST_BASE_URL, model="unsupported-qwen")
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        LLMConfig.from_env({"LLM_BASE_URL": _TEST_BASE_URL})
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        LLMConfig.from_env({"LLM_API_KEY": "x"})


def test_reasoning_content_and_tool_calls_are_normalized_with_mock_transport() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_content": "bounded reasoning",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "inspect_state",
                                        "arguments": '{"month":1}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                "model": APPROVED_MODEL,
            },
        )

    async def scenario() -> LLMResponse:
        transport = _http_client(handler)
        try:
            client = ExternalQwenClient(_config(), http_client=transport)
            return await client.chat(
                [ChatMessage("user", "inspect")],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "inspect_state",
                            "description": "inspect",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                tool_choice="auto",
            )
        finally:
            await transport.aclose()

    response = asyncio.run(scenario())
    assert response.reasoning == "bounded reasoning"
    assert response.tool_calls[0].arguments == {"month": 1}
    assert response.usage.total_tokens == 14
    assert captured["payload"]["model"] == APPROVED_MODEL
    assert captured["payload"]["temperature"] == 0.0
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured["authorization"] == "Bearer test-only-key"


def test_structured_output_disables_thinking_and_uses_json_schema() -> None:
    captured: dict[str, Any] = {}
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"value":7}'}, "finish_reason": "stop"}
                ],
                "model": APPROVED_MODEL,
            },
        )

    async def scenario() -> dict[str, Any]:
        transport = _http_client(handler)
        try:
            client = ExternalQwenClient(_config(), http_client=transport)
            result, _ = await client.structured(
                [ChatMessage("user", "return JSON")], schema=schema, schema_name="Output"
            )
            return result
        finally:
            await transport.aclose()

    assert asyncio.run(scenario()) == {"value": 7}
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["response_format"]["json_schema"] == {
        "name": "Output",
        "strict": True,
        "schema": schema,
    }


def test_provider_error_fails_closed() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    async def scenario() -> None:
        transport = _http_client(handler)
        try:
            client = ExternalQwenClient(_config(), http_client=transport)
            with pytest.raises(LLMError, match="external Qwen request failed"):
                await client.chat([ChatMessage("user", "do not fall back")])
        finally:
            await transport.aclose()

    asyncio.run(scenario())


def test_response_model_mismatch_fails_closed() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "unexpected"}, "finish_reason": "stop"}],
                "model": "different-model",
            },
        )

    async def scenario() -> None:
        transport = _http_client(handler)
        try:
            client = ExternalQwenClient(_config(), http_client=transport)
            with pytest.raises(LLMError, match="model mismatch"):
                await client.chat([ChatMessage("user", "verify")])
        finally:
            await transport.aclose()

    asyncio.run(scenario())


class _WorkflowLLM:
    def __init__(self) -> None:
        self.roles: list[str] = []

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
        del messages, reasoning, tool_choice, max_tokens, timeout_seconds
        calls = (
            (ToolCall("inspect-1", "inspect_state", {"month": 1}),) if tools else ()
        )
        return LLMResponse("preliminary", None, "tool_calls" if calls else "stop", calls)

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
        self.roles.append(role)
        return (
            {
                "role": role,
                "summary": f"summary {role}",
                "recommendation": f"recommendation {role}",
                "evidence": ["deterministic evidence"],
                "approved": True,
            },
            LLMResponse("{}", None, "stop"),
        )


def test_four_role_workflow_is_fixed_and_tools_are_allow_listed() -> None:
    tool_calls: list[Mapping[str, Any]] = []

    def inspect_state(
        arguments: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        tool_calls.append(arguments)
        return {"ok": True, "pressure_atm": 150.0, "track": context["track"]}

    registry = ToolRegistry(
        [
            ToolDefinition(
                "inspect_state",
                "Return validated reservoir state",
                {
                    "type": "object",
                    "properties": {"month": {"type": "integer"}},
                    "required": ["month"],
                    "additionalProperties": False,
                },
                inspect_state,
            )
        ]
    )
    llm = _WorkflowLLM()
    workflow = AgentWorkflow(
        llm,
        registry,
        role_tools={AgentRole.COORDINATOR: ("inspect_state",)},
    )
    state = asyncio.run(workflow.run({"track": 2, "month": "2014-01"}))

    assert tuple(decision.role for decision in state.decisions) == ROLE_ORDER
    assert tuple(llm.roles) == tuple(role.value for role in ROLE_ORDER)
    assert state.complete and state.critic_approved
    assert tool_calls == [{"month": 1}]
    assert state.decisions[0].tool_evidence[0].output["track"] == 2
    assert workflow.transitions == tuple(pairwise(ROLE_ORDER))

    with pytest.raises(ToolNotAllowedError):
        asyncio.run(
            registry.execute(
                ToolCall("bad", "write_schedule", {}), allowed=("inspect_state",)
            )
        )

    with pytest.raises(WorkflowError, match="sensitive key"):
        asyncio.run(workflow.run({"track": 2, "access_token": "test-only"}))


def _chdd_row(date_value: str, well: str, *, producer: bool) -> dict[str, Any]:
    days = 31.0 if date_value.endswith("01-01") else 28.0
    liquid = 10.0 * days if producer else 0.0
    oil = 5.0 * days if producer else 0.0
    injection = 10.0 * days if not producer else 0.0
    return {
        "DATA": date_value,
        "well": well,
        "WLPT": liquid,
        "WLPR": 10.0 if producer else 0.0,
        "WOMT": oil,
        "WOMR": 5.0 if producer else 0.0,
        "WWIR": 0.0 if producer else 10.0,
        "WWIT": injection,
        "THP": 150.0,
        "BHP": 60.0 if producer else 180.0,
        "WEFF": 1.0,
        "WLPT_Diff": liquid,
        "WOMT_Diff": oil,
        "WWIT_Diff": injection,
    }


def test_chdd_adapter_writes_normalized_input_and_manifest(tmp_path: Path) -> None:
    records = [
        _chdd_row("2014-01-01", "1", producer=True),
        _chdd_row("2014-01-01", "101", producer=False),
        _chdd_row("2014-02-01", "1", producer=True),
        _chdd_row("2014-02-01", "101", producer=False),
    ]
    run_dir = tmp_path / "official-economics"
    result = CHDDEconomicsAdapter(timeout_seconds=30).calculate(
        records, start_year=2014, output_dir=run_dir
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.start_date.startswith("2014-")
    assert manifest["start_year"] == 2014
    assert manifest["fields"] == list(CHDD_FIELDS)
    assert manifest["row_count"] == 4
    assert len(manifest["input_sha256"]) == 64
    assert {path.name for path in run_dir.iterdir()} == {
        "input.csv",
        "result.json",
        "report.xlsx",
        "manifest.json",
    }
    with pytest.raises(FileExistsError):
        CHDDEconomicsAdapter().calculate(records, start_year=2014, output_dir=run_dir)


def test_chdd_adapter_records_initial_pump_profile(tmp_path: Path) -> None:
    run_dir = tmp_path / "organizer-reference-economics"
    records = [_chdd_row("2014-01-01", "1", producer=True)]
    adapter = CHDDEconomicsAdapter(timeout_seconds=30)
    result = adapter.calculate(
        records,
        start_year=2014,
        output_dir=run_dir,
        charge_initial_pump=True,
    )
    repeated = adapter.calculate(
        records,
        start_year=2014,
        output_dir=tmp_path / "organizer-reference-economics-repeat",
        charge_initial_pump=True,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    repeated_manifest = json.loads(repeated.manifest_path.read_text(encoding="utf-8"))
    calculation = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))

    assert calculation["assumptions"]["chargeInitialPump"] is True
    assert calculation["summary"]["pumpChanges"] == 1
    assert manifest["assumption_overrides"] == {"chargeInitialPump": True}
    assert manifest["norms_source_sha256"] != manifest["norms_sha256"]
    assert manifest["norms_sha256"] == repeated_manifest["norms_sha256"]
    assert manifest["artifacts"]["effective_norms"] == "norms-effective.xlsx"
    with ZipFile(run_dir / "norms-effective.xlsx") as archive:
        core = archive.read("docProps/core.xml")
    assert b"<dcterms:modified" in core
    assert b">2000-01-01T00:00:00Z</dcterms:modified>" in core
