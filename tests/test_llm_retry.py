import asyncio
import json
from hashlib import sha256
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from timesoil.aios.llm import (
    APPROVED_MODEL,
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
