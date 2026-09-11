"""Fail-closed client for approved Qwen models through external APIs."""

from __future__ import annotations

import asyncio
from email.utils import parsedate_to_datetime
from hashlib import sha256
from ipaddress import ip_address
import json
from math import isfinite
import os
from pathlib import Path
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Self
from urllib.parse import urlsplit

import httpx

APPROVED_MODEL = "qwen3.8-27b"
CEREBRAS_MODEL = "qwen-3.8-27b"
APPROVED_MODELS = frozenset({APPROVED_MODEL, CEREBRAS_MODEL, "qwen3.6-35b-a3b"})
REASONING_EFFORTS = ("none", "low", "medium", "high")
_MAX_REASONING_CHARS = 32_768
_MAX_CONTENT_CHARS = 65_536
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LLMError(RuntimeError):
    """External Qwen generation failed or returned an invalid response."""


def _cerebras_json_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Use agent_rag's Cerebras dialect; role parsers retain local constraints."""
    unsupported = {"pattern", "format", "minItems", "maxItems", "minLength", "maxLength", "uniqueItems", "discriminator"}

    def walk(value: Any, *, names: bool = False) -> Any:
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if names:
                    result[key] = walk(item)
                elif key == "const":
                    result["enum"] = [item]
                elif key not in unsupported:
                    result["anyOf" if key == "oneOf" else key] = walk(
                        item, names=key in {"properties", "$defs", "definitions"}
                    )
            return result
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    return walk(dict(schema))


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Runtime settings; only the approved remote model is accepted."""

    api_key: str = field(repr=False)
    base_url: str
    model: str = APPROVED_MODEL
    timeout_seconds: float = 60.0
    max_output_tokens: int = 4096
    proxy_url: str | None = None
    reasoning_effort: Literal["none", "low", "medium", "high"] = "high"
    seed: int | None = None
    presence_penalty: float = 0.0
    call_log: Path | None = None
    replay_dir: Path | None = None
    fallback: LLMConfig | None = None

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        hostname = parsed.hostname or ""
        if (
            parsed.scheme != "https"
            or not hostname
            or parsed.port not in (None, 443)
            or parsed.path.rstrip("/") != "/v1"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("LLM_BASE_URL must be an external HTTPS /v1 endpoint")
        normalized_host = hostname.rstrip(".").lower()
        try:
            address = ip_address(normalized_host)
        except ValueError:
            if (
                "." not in normalized_host
                or normalized_host == "localhost"
                or normalized_host.endswith(".localhost")
            ):
                raise ValueError("LLM_BASE_URL must use an external host") from None
        else:
            if not address.is_global:
                raise ValueError("LLM_BASE_URL must use a global address")
        if self.model not in APPROVED_MODELS:
            raise ValueError("LLM_MODEL must be an approved external Qwen model")
        if self.model == CEREBRAS_MODEL and self.base_url.rstrip("/") != "https://api.cerebras.ai/v1":
            raise ValueError("Cerebras Qwen requires https://api.cerebras.ai/v1")
        if self.model == APPROVED_MODEL and self.base_url.rstrip("/") != "https://litellm.tatneft.guru/v1":
            raise ValueError("Tatneft Qwen requires https://litellm.tatneft.guru/v1")
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
        if self.proxy_url is not None:
            proxy = urlsplit(self.proxy_url)
            if (
                proxy.scheme not in {"http", "https"}
                or not proxy.hostname
                or proxy.username is not None
                or proxy.password is not None
                or proxy.path not in {"", "/"}
                or proxy.query
                or proxy.fragment
            ):
                raise ValueError("LLM_PROXY_URL must be an HTTP(S) proxy without credentials or path")
            proxy.port  # Validate the optional port before constructing the transport.
        if self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError("LLM_REASONING_EFFORT must be none, low, medium or high")
        if self.seed is not None and not 0 <= self.seed < 2 ** 63:
            raise ValueError("LLM_SEED must be in [0, 2**63)")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("LLM_PRESENCE_PENALTY must be in [-2, 2]")
        if self.call_log is not None and not isinstance(self.call_log, Path):
            raise TypeError("LLM_CALL_LOG must be a path")
        if self.replay_dir is not None and not self.replay_dir.is_dir():
            raise ValueError("LLM_REPLAY_DIR must be an existing directory")
        if self.fallback is not None:
            # One alternate route only: a chain would make the number of remote calls unbounded.
            if not isinstance(self.fallback, LLMConfig) or self.fallback.fallback is not None:
                raise ValueError("LLM fallback must be a single route without its own fallback")
            # Normalise as the other endpoint checks do: a trailing slash or a different case
            # must not disguise the primary endpoint as its own fallback.
            if self.fallback.base_url.rstrip("/").lower() == self.base_url.rstrip("/").lower():
                raise ValueError("LLM_FALLBACK_BASE_URL must differ from LLM_BASE_URL")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> LLMConfig:
        source = os.environ if environ is None else environ
        config = cls(
            api_key=source.get("LLM_API_KEY", ""),
            base_url=source.get("LLM_BASE_URL", ""),
            model=source.get("LLM_MODEL", APPROVED_MODEL),
            timeout_seconds=_env_float(source, "LLM_TIMEOUT_SECONDS", 60.0),
            max_output_tokens=_env_int(source, "LLM_MAX_OUTPUT_TOKENS", 4096),
            proxy_url=source.get("LLM_PROXY_URL") or None,
            reasoning_effort=source.get("LLM_REASONING_EFFORT") or "high",  # type: ignore[arg-type]
            seed=_env_int(source, "LLM_SEED", 0) if source.get("LLM_SEED", "").strip() else None,
            presence_penalty=_env_float(source, "LLM_PRESENCE_PENALTY", 0.0),
            call_log=Path(source["LLM_CALL_LOG"]) if source.get("LLM_CALL_LOG") else None,
            replay_dir=Path(source["LLM_REPLAY_DIR"]) if source.get("LLM_REPLAY_DIR") else None,
        )
        fallback_url = (source.get("LLM_FALLBACK_BASE_URL") or "").strip()
        if not fallback_url:
            return config
        key_file = (source.get("LLM_FALLBACK_API_KEY_FILE") or "").strip()
        if key_file:
            try:
                # A misconfigured key path is a configuration error, not a crash: callers
                # (api.get_agent_workflow, cli) map ValueError to "Qwen is not configured".
                fallback_key = Path(key_file).read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise ValueError("LLM_FALLBACK_API_KEY_FILE is not readable") from exc
        else:
            fallback_key = source.get("LLM_FALLBACK_API_KEY", "")
        # The alternate route inherits every determinism field (seed, temperature, reasoning
        # effort, token budget) and the same call log, so only the endpoint identity differs.
        fallback = replace(
            config,
            api_key=fallback_key,
            base_url=fallback_url,
            model=source.get("LLM_FALLBACK_MODEL") or APPROVED_MODEL,
            # Only the alternate endpoint may need an egress proxy (api.cerebras.ai is geo-blocked).
            proxy_url=source.get("LLM_FALLBACK_PROXY_URL") or config.proxy_url,
        )
        return replace(config, fallback=fallback)


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
    reasoning_tokens: int = 0
    cached_tokens: int = 0


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    reasoning: str | None
    finish_reason: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: LLMUsage = LLMUsage()
    model: str | None = None
    system_fingerprint: str | None = None
    time_info: dict[str, Any] | None = None
    content_sha256: str = ""
    reasoning_sha256: str | None = None


class ExternalQwenClient:
    """Small OpenAI-compatible client: approved remote routes only, never a local model."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        fallback_http_client: httpx.AsyncClient | None = None,
        route: Literal["primary", "fallback"] = "primary",
    ) -> None:
        if http_client is not None and http_client.follow_redirects:
            raise ValueError("LLM transport must not follow redirects")
        if http_client is not None and str(http_client.base_url).rstrip("/") != config.base_url:
            raise ValueError("LLM transport base URL must match configured endpoint")
        if route not in {"primary", "fallback"}:
            raise ValueError("LLM route must be primary or fallback")
        self.config = config
        self.route = route
        self.fallback_answers = 0  # How many calls the alternate route answered; receipt evidence.
        self._fallback = (
            None
            if config.fallback is None
            else ExternalQwenClient(config.fallback, http_client=fallback_http_client, route="fallback")
        )
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=config.base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=httpx.Timeout(config.timeout_seconds),
            trust_env=False,
            proxy=config.proxy_url,
            follow_redirects=False,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._fallback is not None:
            await self._fallback.aclose()
        if self._owns_client:
            await self._client.aclose()

    def fallback_evidence(self) -> dict[str, Any] | None:
        """Receipt provenance: the alternate route's model and how many calls it answered."""
        if self.config.fallback is None:
            return None
        return {"model": self.config.fallback.model, "answered_calls": self.fallback_answers}

    async def _route[T](self, call: Callable[[ExternalQwenClient], Awaitable[T]]) -> T:
        """Run one logical call on the primary and, only on its LLMError, once on the fallback."""
        try:
            return await call(self)
        except LLMError as primary_error:
            if self._fallback is None:
                raise
            try:
                answer = await call(self._fallback)
            except LLMError as fallback_error:
                # Keep the primary's message first: agents.py recovers from its HTTP 400
                # "tool_choice = required" refusal by matching that text.
                raise LLMError(
                    f"{primary_error}; fallback route also failed: {fallback_error}"
                ) from primary_error
            self.fallback_answers += 1
            return answer

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
        return await self._route(
            lambda route: route._chat(
                messages, reasoning=reasoning, tools=tools, tool_choice=tool_choice,
                max_tokens=max_tokens, timeout_seconds=timeout_seconds,
            )
        )

    async def structured(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: Mapping[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[dict[str, Any], LLMResponse]:
        return await self._route(
            lambda route: route._structured(
                messages, schema=schema, schema_name=schema_name,
                max_tokens=max_tokens, timeout_seconds=timeout_seconds,
            )
        )

    async def _chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        reasoning: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMResponse:
        """One route, its own bounded retries; the caller adds the fallback route."""
        payload = self._base_payload(messages, max_tokens=max_tokens)
        if self.config.model == CEREBRAS_MODEL:
            payload["reasoning_effort"] = self.config.reasoning_effort if reasoning else "none"
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": reasoning}
        if tools is not None:
            payload["tools"] = list(tools)
        if tool_choice is not None:
            if tools is None:
                raise LLMError("tool_choice requires tools")
            payload["tool_choice"] = tool_choice
        return await self._post(payload, timeout_seconds=timeout_seconds)

    async def _structured(
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
        if self.config.model == CEREBRAS_MODEL:
            payload["reasoning_effort"] = self.config.reasoning_effort
            schema = _cerebras_json_schema(schema)
        else:
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
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.wire() for message in messages],
            "temperature": 0.0,
        }
        if self.config.model != CEREBRAS_MODEL:
            payload["max_tokens"] = limit
            return payload
        payload["max_completion_tokens"] = limit  # Cerebras rejects max_tokens.
        payload["presence_penalty"] = self.config.presence_penalty
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        return payload

    async def _post(
        self,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float | None,
    ) -> LLMResponse:
        timeout = self.config.timeout_seconds if timeout_seconds is None else timeout_seconds
        if not 0 < timeout <= self.config.timeout_seconds:
            raise ValueError("request timeout must fit configured timeout")
        if "tools" in payload and "response_format" in payload:
            raise LLMError("tools and response_format must not be combined")
        request_sha256 = _sha256_json(payload)  # The payload never carries the API key.
        messages_sha256 = _sha256_json(payload["messages"])
        if self.config.replay_dir is not None:
            return self._replay(request_sha256)
        try:
            async with asyncio.timeout(timeout):
                for attempt in range(5):
                    delay = float(2 ** attempt)
                    started = time.monotonic()
                    try:
                        response = await self._client.post(
                            "chat/completions", json=payload,
                            headers={"Accept-Encoding": "identity",
                                     "Authorization": f"Bearer {self.config.api_key}"},
                        )
                    except httpx.TransportError as exc:
                        self._log_attempt(
                            request_sha256, messages_sha256, attempt,
                            time.monotonic() - started, None, type(exc).__name__,
                        )
                        if attempt == 4:
                            raise
                    else:
                        self._log_attempt(
                            request_sha256, messages_sha256, attempt,
                            time.monotonic() - started, response, None,
                        )
                        if response.status_code not in (408, 429, 500, 502, 503, 504) or attempt == 4:
                            break
                        if response.status_code == 429:
                            delay = float(15 * 2 ** attempt)
                        retry_after = response.headers.get("retry-after")
                        if retry_after:
                            try:
                                requested = float(retry_after)
                            except ValueError:
                                try:
                                    requested = parsedate_to_datetime(retry_after).timestamp() - time.time()
                                except (TypeError, ValueError, OverflowError):
                                    requested = delay
                            if isfinite(requested) and requested >= 0:
                                delay = requested
                    await asyncio.sleep(delay)
            response.raise_for_status()
            body = response.json()
            result = _parse_response(body)
            if result.model != self.config.model:
                raise LLMError("external Qwen response model mismatch")
            return result
        except (TimeoutError, httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            reason = f"HTTP {exc.response.status_code}" if isinstance(exc, httpx.HTTPStatusError) else type(exc).__name__
            raise LLMError(f"external Qwen request failed: {reason}") from exc

    def _replay(self, request_sha256: str) -> LLMResponse:
        """Return the recorded answer for this exact payload; never touch the network."""
        assert self.config.replay_dir is not None
        path = self.config.replay_dir / f"{request_sha256}.json"
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise LLMError(f"replay entry missing for request {request_sha256}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"replay entry is not valid JSON: {path.name}") from exc
        try:
            result = _parse_response(body)
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMError(f"replay entry is not a usable response: {path.name}") from exc
        if result.model != self.config.model:
            raise LLMError("replayed Qwen response model mismatch")
        return result

    def _log_attempt(
        self,
        request_sha256: str,
        messages_sha256: str,
        attempt: int,
        latency_seconds: float,
        response: httpx.Response | None,
        error: str | None,
    ) -> None:
        """Append one audit line per attempt; the key and headers are never written."""
        if self.config.call_log is None:
            return
        body: Any = None
        if response is not None:
            try:
                body = response.json()
            except ValueError:
                body = None
        record: dict[str, Any] = {
            "request_sha256": request_sha256,
            "messages_sha256": messages_sha256,
            "attempt": attempt,
            "http_status": None if response is None else response.status_code,
            "error": error,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "route": self.route,  # Which route answered: "primary" or "fallback".
            "latency_seconds": round(latency_seconds, 6),
            "response_sha256": None if response is None else sha256(response.content).hexdigest(),
            "content_sha256": None,
            "reasoning_sha256": None,
            "tool_calls_sha256": None,
            "system_fingerprint": None,
            "finish_reason": None,
            "usage": None,
            "time_info": None,
        }
        if isinstance(body, Mapping):
            record.update(_audit_fields(body))
            record["response"] = body  # Source for export_replay_dir.
        self.config.call_log.parent.mkdir(parents=True, exist_ok=True)
        with self.config.call_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


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
    time_info = body.get("time_info")
    return LLMResponse(
        content=content,
        reasoning=None if reasoning is None else reasoning[:_MAX_REASONING_CHARS],
        finish_reason=choice.get("finish_reason") if isinstance(choice, Mapping) else None,
        tool_calls=tool_calls,
        usage=_parse_usage(body.get("usage")),
        model=body.get("model") if isinstance(body.get("model"), str) else None,
        system_fingerprint=body.get("system_fingerprint") if isinstance(body.get("system_fingerprint"), str) else None,
        time_info=dict(time_info) if isinstance(time_info, Mapping) else None,
        content_sha256=_sha256_text(content),
        reasoning_sha256=None if reasoning is None else _sha256_text(reasoning),
    )


def _audit_fields(body: Mapping[str, Any]) -> dict[str, Any]:
    """Best-effort hashes of any response body, including error bodies."""
    choices = body.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else {}
    message = choice.get("message")
    message = message if isinstance(message, Mapping) else {}
    raw_content = message.get("content")
    reasoning = _reasoning_text(message)
    tool_calls = message.get("tool_calls")
    return {
        "content_sha256": _sha256_text(raw_content.strip() if isinstance(raw_content, str) else ""),
        "reasoning_sha256": None if reasoning is None else _sha256_text(reasoning),
        "tool_calls_sha256": None if tool_calls is None else _sha256_json(tool_calls),
        "system_fingerprint": body.get("system_fingerprint"),
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"),
        "time_info": body.get("time_info"),
    }


def export_replay_dir(call_log: Path, replay_dir: Path) -> int:
    """Write <request_sha256>.json per logged body so a rerun can use LLM_REPLAY_DIR."""
    replay_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for line in call_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        body = record.get("response")
        request_sha256 = record.get("request_sha256")
        if body is None or not isinstance(request_sha256, str) or not _SHA256_RE.fullmatch(request_sha256):
            continue
        (replay_dir / f"{request_sha256}.json").write_text(
            json.dumps(body, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        written += 1
    return written


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
    """Full reasoning; callers truncate for display, hashes cover the whole text."""
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_usage(raw: Any) -> LLMUsage:
    if not isinstance(raw, Mapping):
        return LLMUsage()

    def count(source: Any, key: str) -> int:
        value = source.get(key, 0) if isinstance(source, Mapping) else 0
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

    completion_details = raw.get("completion_tokens_details")
    prompt_details = raw.get("prompt_tokens_details")
    return LLMUsage(
        count(raw, "prompt_tokens"),
        count(raw, "completion_tokens"),
        count(raw, "total_tokens"),
        count(completion_details, "reasoning_tokens") or count(raw, "reasoning_tokens"),
        count(prompt_details, "cached_tokens") or count(raw, "cached_tokens"),
    )


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


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
