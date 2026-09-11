import asyncio
import json
from hashlib import sha256
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from timesoil.aios.llm import (
    APPROVED_MODEL,
    CEREBRAS_MODEL,
    ChatMessage,
    ExternalQwenClient,
    LLMConfig,
    LLMError,
    export_replay_dir,
)

_BASE_URL = "https://litellm.tatneft.guru/v1"


@pytest.mark.parametrize("statuses, expected_calls", [([0, 429, 200], 3), ([429] * 5, 5), ([401], 1)])
def test_transient_retries_preserve_payload_and_stop_on_fatal_errors(statuses, expected_calls):
    requests = []

    def handler(request):
        requests.append(request.content)
        status = statuses[len(requests) - 1]
        if status == 0:
            raise httpx.ConnectError("temporary", request=request)
        return httpx.Response(status, headers={"Retry-After": "0"}, json={
            "model": APPROVED_MODEL,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        })

    async def run():
        config = LLMConfig(api_key="test-only", base_url="https://litellm.tatneft.guru/v1", timeout_seconds=1)
        async with httpx.AsyncClient(base_url=config.base_url, transport=httpx.MockTransport(handler)) as transport:
            client = ExternalQwenClient(config, http_client=transport)
            return await client.chat([ChatMessage("user", "same request")])

    with patch("timesoil.aios.llm.asyncio.sleep", new_callable=AsyncMock) as sleep:
        if statuses[-1] == 200:
            assert asyncio.run(run()).content == "ok"
            assert [call.args[0] for call in sleep.call_args_list] == [1, 0]
        else:
            with pytest.raises(LLMError):
                asyncio.run(run())
    assert len(requests) == expected_calls and len(set(requests)) == 1


def test_retry_after_cannot_extend_total_request_deadline():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "60"}, json={})

    async def run():
        config = LLMConfig(api_key="test-only", base_url="https://litellm.tatneft.guru/v1", timeout_seconds=.01)
        async with httpx.AsyncClient(base_url=config.base_url, transport=httpx.MockTransport(handler)) as transport:
            with pytest.raises(LLMError, match="TimeoutError"):
                await ExternalQwenClient(config, http_client=transport).chat([ChatMessage("user", "bounded")])

    asyncio.run(run())
    assert calls == 1


def test_rate_limit_retries_can_cross_a_minute_window():
    calls = []

    def handler(request):
        calls.append(request.content)
        return httpx.Response(429 if len(calls) < 4 else 200, json={
            "model": APPROVED_MODEL,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        })

    async def run():
        config = LLMConfig(api_key="test-only", base_url="https://litellm.tatneft.guru/v1")
        async with httpx.AsyncClient(base_url=config.base_url, transport=httpx.MockTransport(handler)) as transport:
            return await ExternalQwenClient(config, http_client=transport).chat([ChatMessage("user", "bounded retry")])

    with patch("timesoil.aios.llm.asyncio.sleep", new_callable=AsyncMock) as sleep:
        assert asyncio.run(run()).content == "ok"
        assert [call.args[0] for call in sleep.call_args_list] == [15, 30, 60]
    assert len(calls) == 4 and len(set(calls)) == 1


def _answer(content: str) -> dict:
    return {
        "model": APPROVED_MODEL,
        "system_fingerprint": "fp_1",
        "choices": [{"message": {"content": content, "reasoning_content": "why"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def test_every_attempt_is_logged_without_the_api_key(tmp_path: Path) -> None:
    call_log = tmp_path / "logs" / "llm_calls.jsonl"
    statuses = [429, 200]
    seen = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        status = statuses[min(seen, len(statuses) - 1)]
        seen += 1
        return httpx.Response(status, headers={"Retry-After": "0"},
                              json={"error": "slow down"} if status == 429 else _answer("ok"))

    async def run():
        config = LLMConfig(api_key="test-only-key", base_url=_BASE_URL, timeout_seconds=2, call_log=call_log)
        async with httpx.AsyncClient(base_url=config.base_url + "/", transport=httpx.MockTransport(handler)) as t:
            return await ExternalQwenClient(config, http_client=t).chat([ChatMessage("user", "audit me")])

    with patch("timesoil.aios.llm.asyncio.sleep", new_callable=AsyncMock):
        response = asyncio.run(run())

    raw = call_log.read_text(encoding="utf-8")
    assert "test-only-key" not in raw and "Bearer" not in raw and "authorization" not in raw.lower()
    records = [json.loads(line) for line in raw.splitlines()]
    assert [record["attempt"] for record in records] == [0, 1]
    assert [record["http_status"] for record in records] == [429, 200]
    assert len({record["request_sha256"] for record in records}) == 1
    assert len({record["response_sha256"] for record in records}) == 2
    assert records[0]["request_sha256"] != records[0]["messages_sha256"]
    assert records[1]["content_sha256"] == response.content_sha256 == sha256(b"ok").hexdigest()
    assert records[1]["reasoning_sha256"] == response.reasoning_sha256
    assert records[1]["system_fingerprint"] == "fp_1" and records[1]["finish_reason"] == "stop"
    assert records[1]["usage"]["total_tokens"] == 5 and records[1]["tool_calls_sha256"] is None
    assert records[1]["model"] == APPROVED_MODEL and records[1]["base_url"] == _BASE_URL
    assert isinstance(records[1]["latency_seconds"], float)


def test_replay_returns_the_recorded_answer_and_fails_on_a_miss(tmp_path: Path) -> None:
    call_log = tmp_path / "llm_calls.jsonl"
    replay_dir = tmp_path / "replay"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_answer("recorded"))

    def refuse(_: httpx.Request) -> httpx.Response:
        raise AssertionError("replay must not reach the network")

    async def record():
        config = LLMConfig(api_key="test-only-key", base_url=_BASE_URL, timeout_seconds=2, call_log=call_log)
        async with httpx.AsyncClient(base_url=config.base_url + "/", transport=httpx.MockTransport(handler)) as t:
            return await ExternalQwenClient(config, http_client=t).chat([ChatMessage("user", "replay me")])

    async def replay(prompt: str):
        config = LLMConfig(api_key="test-only-key", base_url=_BASE_URL, timeout_seconds=2, replay_dir=replay_dir)
        async with httpx.AsyncClient(base_url=config.base_url + "/", transport=httpx.MockTransport(refuse)) as t:
            return await ExternalQwenClient(config, http_client=t).chat([ChatMessage("user", prompt)])

    original = asyncio.run(record())
    assert export_replay_dir(call_log, replay_dir) == 1
    assert [path.stem for path in replay_dir.iterdir()] == [json.loads(call_log.read_text())["request_sha256"]]

    replayed = asyncio.run(replay("replay me"))
    assert replayed.content == original.content == "recorded"
    assert replayed.content_sha256 == original.content_sha256
    with pytest.raises(LLMError, match="replay entry missing"):
        asyncio.run(replay("never recorded"))


_FALLBACK_URL = "https://api.cerebras.ai/v1"
_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def _route_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    key_file = tmp_path / "fallback-key"
    key_file.write_text("fallback-test-only-key\n", encoding="utf-8")
    return {
        "LLM_API_KEY": "primary-test-only-key",
        "LLM_BASE_URL": _BASE_URL,
        "LLM_TIMEOUT_SECONDS": "2",
        "LLM_SEED": "20260909",
        "LLM_CALL_LOG": str(tmp_path / "llm_calls.jsonl"),
        "LLM_FALLBACK_BASE_URL": _FALLBACK_URL,
        "LLM_FALLBACK_MODEL": CEREBRAS_MODEL,
        "LLM_FALLBACK_API_KEY_FILE": str(key_file),
        **extra,
    }


def _two_route_client(config: LLMConfig, primary, fallback) -> ExternalQwenClient:
    assert config.fallback is not None
    return ExternalQwenClient(
        config,
        http_client=httpx.AsyncClient(base_url=config.base_url + "/", transport=httpx.MockTransport(primary)),
        fallback_http_client=httpx.AsyncClient(
            base_url=config.fallback.base_url + "/", transport=httpx.MockTransport(fallback)
        ),
    )


def test_fallback_route_answers_when_the_primary_structured_reply_is_incomplete(tmp_path: Path) -> None:
    config = LLMConfig.from_env(_route_env(tmp_path))
    assert config.fallback is not None and config.fallback.api_key == "fallback-test-only-key"
    assert config.fallback.seed == config.seed == 20260909  # Determinism fields are inherited.
    primary_bodies: list[bytes] = []
    fallback_bodies: list[bytes] = []

    def primary(request: httpx.Request) -> httpx.Response:
        primary_bodies.append(request.content)
        return httpx.Response(200, json={  # Truncated answer: the incomplete-structured guard fires.
            "model": APPROVED_MODEL,
            "choices": [{"message": {"content": '{"ok": tr'}, "finish_reason": "length"}],
        })

    def fallback(request: httpx.Request) -> httpx.Response:
        fallback_bodies.append(request.content)
        return httpx.Response(200, json={
            "model": CEREBRAS_MODEL,
            "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
        })

    async def run():
        async with _two_route_client(config, primary, fallback) as client:
            return await client.structured(
                [ChatMessage("user", "decide")], schema=_SCHEMA, schema_name="decision"
            )

    payload, response = asyncio.run(run())
    assert payload == {"ok": True} and response.model == CEREBRAS_MODEL
    assert len(primary_bodies) == 1 and len(fallback_bodies) == 1
    assert config.call_log is not None
    raw = config.call_log.read_text(encoding="utf-8")
    records = [json.loads(line) for line in raw.splitlines()]
    assert [record["route"] for record in records] == ["primary", "fallback"]
    assert [record["base_url"] for record in records] == [_BASE_URL, _FALLBACK_URL]
    assert "fallback-test-only-key" not in raw


def test_both_routes_failing_raises_after_each_route_ran_its_own_retries(tmp_path: Path) -> None:
    config = LLMConfig.from_env(_route_env(tmp_path))
    seen: list[str] = []

    def refuse(route: str):
        def handler(_: httpx.Request) -> httpx.Response:
            seen.append(route)
            return httpx.Response(400, json={"error": "Failed to generate tool call"})

        return handler

    async def run():
        async with _two_route_client(config, refuse("primary"), refuse("fallback")) as client:
            return await client.chat([ChatMessage("user", "decide")])

    with pytest.raises(LLMError, match="HTTP 400"):
        asyncio.run(run())
    assert seen == ["primary", "fallback"]  # HTTP 400 is fatal per route, so one attempt each.
    assert config.call_log is not None
    records = [json.loads(line) for line in config.call_log.read_text(encoding="utf-8").splitlines()]
    assert [record["route"] for record in records] == ["primary", "fallback"]


def test_a_fallback_route_must_differ_from_the_primary_and_must_not_chain(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="LLM_FALLBACK_BASE_URL"):
        LLMConfig.from_env(
            _route_env(tmp_path, LLM_FALLBACK_BASE_URL=_BASE_URL, LLM_FALLBACK_MODEL=APPROVED_MODEL)
        )
    chained = LLMConfig.from_env(_route_env(tmp_path))
    with pytest.raises(ValueError, match="without its own fallback"):
        LLMConfig(api_key="test-only-key", base_url=_BASE_URL, fallback=chained)
