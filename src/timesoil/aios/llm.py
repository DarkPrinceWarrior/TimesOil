"""Fail-closed client for Tatneft Qwen3.6 through its LiteLLM API."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Self
from urllib.parse import urlsplit

import httpx

APPROVED_BASE_URL = "https://litellm.tatneft.guru/v1"
APPROVED_MODEL = "qwen3.6-35b-a3b"
_MAX_REASONING_CHARS = 32_768
_MAX_CONTENT_CHARS = 65_536
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")


class LLMError(RuntimeError):
    """Tatneft generation failed or returned an invalid response."""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Runtime settings; only the approved remote model is accepted."""

    api_key: str = field(repr=False)
    base_url: str = APPROVED_BASE_URL
    model: str = APPROVED_MODEL
    timeout_seconds: float = 60.0
    max_output_tokens: int = 4096

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "litellm.tatneft.guru"
            or parsed.port not in (None, 443)
            or parsed.path.rstrip("/") != "/v1"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("LLM_BASE_URL is not the approved Tatneft LiteLLM endpoint")
        if self.model != APPROVED_MODEL:
            raise ValueError("LLM_MODEL must be qwen3.6-35b-a3b")
        if (
            not self.api_key
            or self.api_key != self.api_key.strip()
            or any(ord(character) < 33 or ord(character) == 127 for character in self.api_key)
        ):
            raise ValueError("LLM_API_KEY is required")
        if not 0 < self.timeout_seconds <= 3600:
            raise ValueError("LLM_TIMEOUT_SECONDS must be in (0, 3600]")
        if not 1 <= self.max_output_tokens <= 32_768:
            raise ValueError("LLM_MAX_OUTPUT_TOKENS must be in [1, 32768]")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> LLMConfig:
        source = os.environ if environ is None else environ
        return cls(
            api_key=source.get("LLM_API_KEY", ""),
            base_url=source.get("LLM_BASE_URL", APPROVED_BASE_URL),
            model=source.get("LLM_MODEL", APPROVED_MODEL),
            timeout_seconds=_env_float(source, "LLM_TIMEOUT_SECONDS", 60.0),
            max_output_tokens=_env_int(source, "LLM_MAX_OUTPUT_TOKENS", 4096),
        )


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {self.role}")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")

    def wire(self) -> dict[str, str]:
        payload = {"role": self.role, "content": self.content}
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    reasoning: str | None
    finish_reason: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: LLMUsage = LLMUsage()
    model: str | None = None


class TatneftLLMClient:
    """Small OpenAI-compatible client with no provider or local fallback."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if http_client is not None and http_client.follow_redirects:
            raise ValueError("LLM transport must not follow redirects")
        if http_client is not None and str(http_client.base_url).rstrip("/") != config.base_url:
            raise ValueError("LLM transport base URL must match the approved endpoint")
        self.config = config
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=config.base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=httpx.Timeout(config.timeout_seconds),
            trust_env=False,
            follow_redirects=False,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

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
        payload = self._base_payload(messages, max_tokens=max_tokens)
        payload["chat_template_kwargs"] = {"enable_thinking": reasoning}
        if tools is not None:
            payload["tools"] = list(tools)
        if tool_choice is not None:
            if tools is None:
                raise LLMError("tool_choice requires tools")
            payload["tool_choice"] = tool_choice
        return await self._post(payload, timeout_seconds=timeout_seconds)

    async def structured(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: Mapping[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[dict[str, Any], LLMResponse]:
        if not _NAME_RE.fullmatch(schema_name):
            raise ValueError("invalid JSON schema name")
        payload = self._base_payload(messages, max_tokens=max_tokens)
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": dict(schema)},
        }
        response = await self._post(payload, timeout_seconds=timeout_seconds)
        if response.finish_reason == "length" or not response.content:
            raise LLMError("structured response is incomplete")
        try:
            result = json.loads(_strip_json_fence(response.content))
        except (json.JSONDecodeError, TypeError) as exc:
            raise LLMError("structured response is not valid JSON") from exc
        if not isinstance(result, dict):
            raise LLMError("structured response must be a JSON object")
        return result, response

    def _base_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        if not messages:
            raise ValueError("at least one message is required")
        limit = self.config.max_output_tokens if max_tokens is None else max_tokens
        if not 1 <= limit <= self.config.max_output_tokens:
            raise ValueError("max_tokens exceeds configured output limit")
        return {
            "model": self.config.model,
            "messages": [message.wire() for message in messages],
            "temperature": 0.0,
            "max_tokens": limit,
        }

    async def _post(
        self,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float | None,
    ) -> LLMResponse:
        timeout = self.config.timeout_seconds if timeout_seconds is None else timeout_seconds
        if not 0 < timeout <= self.config.timeout_seconds:
            raise ValueError("request timeout must fit configured timeout")
        try:
            async with asyncio.timeout(timeout):
                response = await self._client.post(
                    "chat/completions",
                    json=payload,
                    headers={
                        "Accept-Encoding": "identity",
                        "Authorization": f"Bearer {self.config.api_key}",
                    },
                )
            response.raise_for_status()
            body = response.json()
            return _parse_response(body)
        except (TimeoutError, httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise LLMError("Tatneft LiteLLM request failed") from exc


def _parse_response(body: Any) -> LLMResponse:
    if not isinstance(body, Mapping):
        raise TypeError("response body is not an object")
    choices = body["choices"]
    if not isinstance(choices, list) or not choices:
        raise TypeError("response choices are missing")
    choice = choices[0]
    message = choice["message"]
    if not isinstance(message, Mapping):
        raise TypeError("response message is not an object")
    raw_content = message.get("content")
    content = raw_content.strip() if isinstance(raw_content, str) else ""
    if len(content) > _MAX_CONTENT_CHARS:
        raise ValueError("response content exceeds the bounded contract")
    reasoning = _reasoning_text(message)
    tool_calls = _parse_tool_calls(message.get("tool_calls"))
    if not content and reasoning is None and not tool_calls:
        raise TypeError("response contains no usable output")
    return LLMResponse(
        content=content,
        reasoning=reasoning,
        finish_reason=choice.get("finish_reason") if isinstance(choice, Mapping) else None,
        tool_calls=tool_calls,
        usage=_parse_usage(body.get("usage")),
        model=body.get("model") if isinstance(body.get("model"), str) else None,
    )


def _parse_tool_calls(raw_calls: Any) -> tuple[ToolCall, ...]:
    if raw_calls is None:
        return ()
    if not isinstance(raw_calls, list):
        raise TypeError("tool_calls must be a list")
    calls: list[ToolCall] = []
    seen: set[str] = set()
    for raw in raw_calls:
        if not isinstance(raw, Mapping) or raw.get("type", "function") != "function":
            raise TypeError("invalid tool call")
        call_id = raw.get("id")
        function = raw.get("function")
        if not isinstance(call_id, str) or not call_id or call_id in seen:
            raise ValueError("invalid or duplicate tool call id")
        if not isinstance(function, Mapping):
            raise TypeError("tool call function is missing")
        name = function.get("name")
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise ValueError("invalid tool name")
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, Mapping):
            raise TypeError("tool arguments must be a JSON object")
        calls.append(ToolCall(id=call_id, name=name, arguments=dict(arguments)))
        seen.add(call_id)
    return tuple(calls)


def _reasoning_text(message: Mapping[str, Any]) -> str | None:
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:_MAX_REASONING_CHARS]
    return None


def _parse_usage(raw: Any) -> LLMUsage:
    if not isinstance(raw, Mapping):
        return LLMUsage()
    values = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = raw.get(key, 0)
        values.append(value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0)
    return LLMUsage(*values)


def _strip_json_fence(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    return "\n".join(lines[1:-1]).strip() if len(lines) >= 3 and lines[-1].strip() == "```" else stripped


def _env_float(source: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(source.get(key, str(default)))
    except ValueError:
        raise ValueError(f"{key} must be numeric") from None


def _env_int(source: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(source.get(key, str(default)))
    except ValueError:
        raise ValueError(f"{key} must be an integer") from None
