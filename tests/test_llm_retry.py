import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from timesoil.aios.llm import APPROVED_MODEL, ChatMessage, ExternalQwenClient, LLMConfig, LLMError


@pytest.mark.parametrize("statuses, expected_calls", [([0, 429, 200], 3), ([429] * 3, 3), ([401], 1)])
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
        config = LLMConfig(api_key="test-only", base_url="https://api.cerebras.ai/v1", timeout_seconds=1)
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
        config = LLMConfig(api_key="test-only", base_url="https://api.cerebras.ai/v1", timeout_seconds=.01)
        async with httpx.AsyncClient(base_url=config.base_url, transport=httpx.MockTransport(handler)) as transport:
            with pytest.raises(LLMError, match="TimeoutError"):
                await ExternalQwenClient(config, http_client=transport).chat([ChatMessage("user", "bounded")])

    asyncio.run(run())
    assert calls == 1
