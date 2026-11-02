from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any
import httpx
import json
import re
import time
from urllib.parse import urlparse

from openai import APIStatusError, OpenAI, RateLimitError

from EmoTrack.conversation.config import ModelConfig
from EmoTrack.conversation.openai_rate_control import (
    estimate_chat_request_tokens,
    get_openai_chat_rate_controller,
)
from EmoTrack.inference.backend import InferenceConfig, build_inference_backend
from EmoTrack.utils.json_utils import extract_json_object
from EmoTrack.utils.openai_compat import prefers_max_completion_tokens
from EmoTrack.utils.vertex_auth import (
    VertexAccessTokenProvider,
    is_vertex_base_url,
    normalize_vertex_api_root,
    normalize_vertex_model_name,
)
from EmoTrack.utils.vertex_gemini import (
    build_vertex_generate_content_payload,
    extract_vertex_generation_result,
)


_THINKING_TAG_RE = re.compile(r"<(?:think|thinking)>(.*?)</(?:think|thinking)>", flags=re.IGNORECASE | re.DOTALL)


def _join_text_parts(parts: list[str]) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip()).strip()


def _preview_text(text: str, *, limit: int = 240) -> str:
    normalized = " ".join((text or "").split()).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _split_thinking_from_text(text: str, *, strip_tags: bool) -> tuple[str, str]:
    raw = (text or "").strip()
    if not raw:
        return "", ""

    matches = [match.strip() for match in _THINKING_TAG_RE.findall(raw) if match.strip()]
    if not strip_tags:
        return raw, _join_text_parts(matches)

    visible = _THINKING_TAG_RE.sub("", raw).strip()
    return visible, _join_text_parts(matches)


def _extract_content_item_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if item is None:
        return ""
    if isinstance(item, (list, tuple)):
        return _join_text_parts([_extract_content_item_text(part) for part in item])
    if isinstance(item, dict):
        for key in ("text", "content", "output_text", "value", "transcript"):
            if key in item:
                return _extract_content_item_text(item.get(key))
        return ""
    for attr in ("text", "content", "output_text", "value", "transcript"):
        value = getattr(item, attr, None)
        if value is None:
            continue
        extracted = _extract_content_item_text(value)
        if extracted:
            return extracted
    return ""


def _base_url_looks_local(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").strip().lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _serialize_parsed_message(parsed: Any) -> str:
    if parsed is None:
        return ""
    if isinstance(parsed, str):
        return parsed.strip()
    if hasattr(parsed, "model_dump"):
        parsed = parsed.model_dump()
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)
    extracted = _extract_content_item_text(parsed)
    if extracted:
        return extracted
    try:
        return json.dumps(parsed, ensure_ascii=False)
    except TypeError:
        return ""


def _extract_visible_and_thinking(message: Any, strip_tags: bool) -> tuple[str, str]:
    raw_thinking_parts: list[str] = []
    visible_parts: list[str] = []

    for attr in ("reasoning_content", "reasoning", "thinking_content"):
        value = getattr(message, attr, None)
        if isinstance(value, str) and value.strip():
            raw_thinking_parts.append(value.strip())

    parsed_text = _serialize_parsed_message(getattr(message, "parsed", None))
    if parsed_text:
        visible_parts.append(parsed_text)

    content = getattr(message, "content", "")
    if isinstance(content, str):
        visible, embedded_thinking = _split_thinking_from_text(content, strip_tags=strip_tags)
        if visible:
            visible_parts.append(visible)
        if embedded_thinking:
            raw_thinking_parts.append(embedded_thinking)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                item_type = str(item.get("type") or "").strip().lower()
            else:
                item_type = str(getattr(item, "type", "") or "").strip().lower()
            text = _extract_content_item_text(item)
            if not text:
                continue
            if item_type in {"reasoning", "thinking"}:
                raw_thinking_parts.append(text)
                continue
            visible, embedded_thinking = _split_thinking_from_text(text, strip_tags=strip_tags)
            if visible:
                visible_parts.append(visible)
            if embedded_thinking:
                raw_thinking_parts.append(embedded_thinking)
    else:
        fallback_text = _extract_content_item_text(content)
        visible, embedded_thinking = _split_thinking_from_text(fallback_text, strip_tags=strip_tags)
        if visible:
            visible_parts.append(visible)
        if embedded_thinking:
            raw_thinking_parts.append(embedded_thinking)

    if not visible_parts:
        for attr in ("output_text", "text"):
            fallback = _extract_content_item_text(getattr(message, attr, None))
            if not fallback:
                continue
            visible, embedded_thinking = _split_thinking_from_text(fallback, strip_tags=strip_tags)
            if visible:
                visible_parts.append(visible)
            if embedded_thinking:
                raw_thinking_parts.append(embedded_thinking)

    return _join_text_parts(visible_parts), _join_text_parts(raw_thinking_parts)


def _extract_visible_and_thinking_from_response(response: Any, *, strip_tags: bool) -> tuple[str, str]:
    raw_thinking_parts: list[str] = []
    visible_parts: list[str] = []

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        visible, embedded_thinking = _split_thinking_from_text(output_text, strip_tags=strip_tags)
        if visible:
            visible_parts.append(visible)
        if embedded_thinking:
            raw_thinking_parts.append(embedded_thinking)

    reasoning = getattr(response, "reasoning", None)
    if isinstance(reasoning, dict):
        summary = reasoning.get("summary")
        if isinstance(summary, str) and summary.strip():
            raw_thinking_parts.append(summary.strip())
        elif isinstance(summary, list):
            for item in summary:
                text = _extract_content_item_text(item)
                if text:
                    raw_thinking_parts.append(text)

    output_items = getattr(response, "output", None)
    if isinstance(output_items, list) and not visible_parts:
        for item in output_items:
            if isinstance(item, dict):
                item_type = str(item.get("type") or "").strip().lower()
                content = item.get("content")
            else:
                item_type = str(getattr(item, "type", "") or "").strip().lower()
                content = getattr(item, "content", None)

            if item_type == "reasoning":
                text = _extract_content_item_text(item)
                if text:
                    raw_thinking_parts.append(text)
                continue

            if item_type != "message" or not isinstance(content, list):
                continue

            for content_item in content:
                if isinstance(content_item, dict):
                    content_type = str(content_item.get("type") or "").strip().lower()
                else:
                    content_type = str(getattr(content_item, "type", "") or "").strip().lower()
                text = _extract_content_item_text(content_item)
                if not text:
                    continue
                if content_type in {"reasoning", "thinking"}:
                    raw_thinking_parts.append(text)
                    continue
                visible, embedded_thinking = _split_thinking_from_text(text, strip_tags=strip_tags)
                if visible:
                    visible_parts.append(visible)
                if embedded_thinking:
                    raw_thinking_parts.append(embedded_thinking)

    return _join_text_parts(visible_parts), _join_text_parts(raw_thinking_parts)


@dataclass
class GenerationResult:
    text: str
    thinking: str
    raw_message: dict[str, Any]


class _EmptyVisibleResponseError(RuntimeError):
    def __init__(self, *, thinking: str, raw_message: dict[str, Any]) -> None:
        self.thinking = thinking
        self.raw_message = raw_message
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        if self.thinking:
            preview = _preview_text(self.thinking)
            if preview:
                return f"Model returned reasoning without a visible answer. Reasoning preview: {preview}"
        return "Model returned reasoning without a visible answer"


class RetryableGenerationError(RuntimeError):
    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ReasoningClient:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._source = (config.source or "api").strip().lower()
        self._client = None
        self._vertex_http: httpx.Client | None = None
        self._backend = None
        self._rate_controller = None
        self._api_key_provider: VertexAccessTokenProvider | None = None
        self._active_api_key = ""
        if self._source == "api":
            if self._is_vertex_mode():
                raise ValueError("Vertex AI endpoints require models.source=vertex, not models.source=api")
            self._api_key_provider = VertexAccessTokenProvider(
                base_url=config.base_url,
                initial_api_key=config.api_key or "EMPTY",
            )
            self._ensure_api_client(force=True)
            self._rate_controller = get_openai_chat_rate_controller(
                base_url=config.base_url,
                model_name=config.model_name,
            )
        elif self._source == "vertex":
            if not is_vertex_base_url(config.base_url):
                raise ValueError("models.source=vertex requires a native Vertex AI base_url")
            normalize_vertex_api_root(config.base_url)
            self._api_key_provider = VertexAccessTokenProvider(
                base_url=config.base_url,
                initial_api_key=config.api_key or "EMPTY",
            )
            self._vertex_http = httpx.Client(timeout=config.timeout_seconds)
            self._rate_controller = get_openai_chat_rate_controller(
                base_url=config.base_url,
                model_name=config.model_name,
            )
        elif self._source == "vllm":
            self._backend = build_inference_backend(
                InferenceConfig(
                    backend="vllm",
                    model=config.model_name,
                    request_mode="quick",
                    vllm=config.vllm,
                )
            )
        else:
            raise ValueError(f"Unsupported model source: {config.source}")

    def _is_vertex_mode(self) -> bool:
        provider = self._api_key_provider
        if provider is not None:
            return provider.enabled
        return is_vertex_base_url(self.config.base_url)

    def _resolve_api_key(self, *, force_refresh: bool = False) -> str:
        if self._api_key_provider is None:
            return str(self.config.api_key or "EMPTY").strip() or "EMPTY"
        token = (
            self._api_key_provider.force_refresh()
            if force_refresh
            else self._api_key_provider.current_api_key()
        )
        return token or "EMPTY"

    def _ensure_api_client(self, *, force: bool = False) -> None:
        api_key = self._resolve_api_key(force_refresh=force)
        if self._client is not None and not force and api_key == self._active_api_key:
            return
        self._active_api_key = api_key
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
        )

    def _build_request_extras(self, *, thinking_enabled: bool | None = None) -> dict[str, Any]:
        enabled = self.config.thinking.enabled if thinking_enabled is None else thinking_enabled
        extra_body = copy.deepcopy(self.config.thinking.request_extras)
        raw_chat_template_kwargs = extra_body.get("chat_template_kwargs")
        explicit_disable = (
            isinstance(raw_chat_template_kwargs, dict)
            and raw_chat_template_kwargs.get("enable_thinking") is False
        )
        if not enabled:
            if isinstance(raw_chat_template_kwargs, dict):
                if explicit_disable:
                    raw_chat_template_kwargs["enable_thinking"] = False
                    extra_body["chat_template_kwargs"] = raw_chat_template_kwargs
                else:
                    raw_chat_template_kwargs.pop("enable_thinking", None)
                    if raw_chat_template_kwargs:
                        extra_body["chat_template_kwargs"] = raw_chat_template_kwargs
                    else:
                        extra_body.pop("chat_template_kwargs", None)
            extra_body.pop("thinking_budget", None)
            return extra_body
        if enabled:
            chat_template_kwargs = raw_chat_template_kwargs if isinstance(raw_chat_template_kwargs, dict) else {}
            chat_template_kwargs.setdefault("enable_thinking", True)
            extra_body["chat_template_kwargs"] = chat_template_kwargs
        if enabled and self.config.thinking.budget_tokens > 0:
            extra_body.setdefault("thinking_budget", self.config.thinking.budget_tokens)
        return extra_body

    @property
    def source(self) -> str:
        return self._source

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        vertex_response_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        if self._source == "vllm":
            extra_body = self._build_request_extras()
            raw, visible, thinking = self._generate_vllm_text(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                extra_body=extra_body,
            )
            return GenerationResult(
                text=visible,
                thinking=thinking,
                raw_message={"content": raw},
            )
        if self._source == "vertex":
            request = self._build_vertex_request(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                thinking_enabled=self.config.thinking.enabled,
                vertex_response_schema=vertex_response_schema,
            )
        else:
            request = self._build_live_api_request(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                thinking_enabled=self.config.thinking.enabled,
            )

        last_error: Exception | None = None
        max_attempts = max(1, self.config.max_retries)
        for attempt in range(1, max_attempts + 1):
            try:
                if self._source == "vertex":
                    result = self._generate_vertex_once(request=request)
                else:
                    result = self._generate_api_transport_once(request=request)
                if self._wants_json_object(response_format):
                    return self._normalize_api_json_result(
                        messages=messages,
                        result=result,
                        max_tokens=max_tokens,
                        vertex_response_schema=vertex_response_schema,
                    )
                return result
            except Exception as exc:
                last_error = exc
                if attempt >= max_attempts:
                    break
                time.sleep(min(attempt, 3))

        # Some providers intermittently return empty visible content even when
        # the request asks for a normal chat or JSON response. Fall back to a
        # plain-response pass and, for JSON-mode calls, repair into a strict object.
        if isinstance(last_error, _EmptyVisibleResponseError):
            try:
                return self._recover_empty_api_response(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    vertex_response_schema=vertex_response_schema,
                )
            except Exception as fallback_error:
                last_error = fallback_error

        if last_error is None:
            raise RuntimeError("Model generation failed")
        if _is_rate_limit_exception(last_error):
            raise RetryableGenerationError(
                f"Model generation rate limited after {max_attempts} attempts: {last_error}",
                retry_after_seconds=_retry_after_seconds_from_exception(last_error),
            ) from last_error
        raise RuntimeError(f"Model generation failed after {max_attempts} attempts: {last_error}") from last_error

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
        if self._vertex_http is not None:
            self._vertex_http.close()

    @staticmethod
    def _request_max_tokens(request: dict[str, Any]) -> int:
        for key in ("max_output_tokens", "max_completion_tokens", "max_tokens", "maxOutputTokens"):
            value = request.get(key)
            if isinstance(value, int):
                return value
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return 0

    @staticmethod
    def _vertex_contents_to_messages(contents: list[dict[str, Any]]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for item in contents:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user").strip().lower()
            parts = item.get("parts")
            if not isinstance(parts, list):
                continue
            text_parts = [
                str(part.get("text") or "").strip()
                for part in parts
                if isinstance(part, dict) and str(part.get("text") or "").strip()
            ]
            if not text_parts:
                continue
            messages.append(
                {
                    "role": "assistant" if role == "model" else "user",
                    "content": _join_text_parts(text_parts),
                }
            )
        return messages

    def _build_vertex_request(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
        thinking_enabled: bool,
        vertex_response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "url": (
                f"{normalize_vertex_api_root(self.config.base_url)}/publishers/google/models/"
                f"{normalize_vertex_model_name(self.config.model_name)}:generateContent"
            ),
            "json": build_vertex_generate_content_payload(
                messages=messages,
                model_name=self.config.model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                response_schema=vertex_response_schema,
                request_extras=self._build_request_extras(thinking_enabled=thinking_enabled),
                thinking_enabled=thinking_enabled,
                thinking_budget_tokens=self.config.thinking.budget_tokens,
            ),
        }

    @staticmethod
    def _sanitize_live_api_extra_body(
        extra_body: dict[str, Any],
        *,
        responses_api: bool,
        base_url: str,
    ) -> dict[str, Any]:
        if _base_url_looks_local(base_url):
            return copy.deepcopy(extra_body)
        sanitized = copy.deepcopy(extra_body)
        # These knobs are only consumed by local vLLM prompting/sampling and are
        # not valid OpenAI API request fields.
        for key in ("chat_template_kwargs", "top_k", "repetition_penalty", "thinking_budget"):
            sanitized.pop(key, None)
        if responses_api:
            sanitized.pop("frequency_penalty", None)
        return sanitized

    def _build_api_request(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
        thinking_enabled: bool,
    ) -> dict[str, Any]:
        is_gpt5_family = prefers_max_completion_tokens(self.config.model_name)
        token_param = (
            "max_completion_tokens"
            if is_gpt5_family
            else "max_tokens"
        )
        request: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            token_param: max_tokens,
        }
        # Current GPT-5 chat-completions endpoints accept only the default temperature.
        if not is_gpt5_family:
            request["temperature"] = temperature
        if response_format is not None and not (
            is_gpt5_family and self._wants_json_object(response_format)
        ):
            request["response_format"] = response_format

        extra_body = self._sanitize_live_api_extra_body(
            self._build_request_extras(thinking_enabled=thinking_enabled),
            responses_api=False,
            base_url=self.config.base_url,
        )
        top_p = extra_body.pop("top_p", None)
        frequency_penalty = extra_body.pop("frequency_penalty", None)
        if top_p is not None:
            request["top_p"] = top_p
        if frequency_penalty is not None:
            request["frequency_penalty"] = frequency_penalty
        if extra_body:
            request["extra_body"] = extra_body
        return request

    def _build_responses_request(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
        thinking_enabled: bool,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.config.model_name,
            "input": messages,
            "max_output_tokens": max_tokens,
        }
        is_gpt5_family = prefers_max_completion_tokens(self.config.model_name)
        if not is_gpt5_family:
            request["temperature"] = temperature
        text_config: dict[str, Any] = {}
        if response_format is not None:
            text_config["format"] = response_format
        if is_gpt5_family:
            request["reasoning"] = {"effort": "none"}
            text_config.setdefault("verbosity", "low")
        if text_config:
            request["text"] = text_config

        extra_body = self._sanitize_live_api_extra_body(
            self._build_request_extras(thinking_enabled=thinking_enabled),
            responses_api=True,
            base_url=self.config.base_url,
        )
        top_p = extra_body.pop("top_p", None)
        frequency_penalty = extra_body.pop("frequency_penalty", None)
        if top_p is not None:
            request["top_p"] = top_p
        if frequency_penalty is not None:
            request["frequency_penalty"] = frequency_penalty
        if extra_body:
            request["extra_body"] = extra_body
        return request

    def _uses_responses_api(self) -> bool:
        return self._source == "api" and prefers_max_completion_tokens(self.config.model_name)

    def _build_live_api_request(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
        thinking_enabled: bool,
    ) -> dict[str, Any]:
        if self._uses_responses_api():
            return self._build_responses_request(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                thinking_enabled=thinking_enabled,
            )
        return self._build_api_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            thinking_enabled=thinking_enabled,
        )

    def _generate_api_transport_once(self, *, request: dict[str, Any]) -> GenerationResult:
        if self._uses_responses_api():
            return self._generate_responses_api_once(request=request)
        return self._generate_api_once(request=request)

    def _generate_api_once(self, *, request: dict[str, Any]) -> GenerationResult:
        self._ensure_api_client()
        response = None
        reservation = None
        if self._rate_controller is not None:
            reservation = self._rate_controller.acquire(
                estimated_tokens=estimate_chat_request_tokens(
                    messages=request["messages"],
                    max_tokens=self._request_max_tokens(request),
                )
            )
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            if _is_auth_exception(exc) and self._api_key_provider is not None:
                self._ensure_api_client(force=True)
                response = self._client.chat.completions.create(**request)
                exc = None
            if exc is not None:
                if self._rate_controller is not None and _is_rate_limit_exception(exc):
                    self._rate_controller.note_rate_limit(
                        retry_after_seconds=_retry_after_seconds_from_exception(exc)
                    )
                raise
        finally:
            if reservation is not None:
                actual_tokens = _response_total_tokens(response) if response is not None else None
                reservation.complete(actual_tokens=actual_tokens)
        if not response.choices:
            raise RuntimeError("Model response has no choices")
        choice = response.choices[0]
        message = choice.message
        usage = getattr(response, "usage", None)
        raw_message = {
            "provider": "openai_chat_completions",
            "model": getattr(response, "model", None),
            "response_id": getattr(response, "id", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
            "message": message.model_dump() if hasattr(message, "model_dump") else {"content": getattr(message, "content", "")},
        }
        visible, thinking = _extract_visible_and_thinking(
            message,
            strip_tags=self.config.thinking.strip_tags,
        )
        if not visible:
            raise _EmptyVisibleResponseError(
                thinking=thinking,
                raw_message=raw_message,
            )
        return GenerationResult(text=visible, thinking=thinking, raw_message=raw_message)

    def _generate_responses_api_once(self, *, request: dict[str, Any]) -> GenerationResult:
        self._ensure_api_client()
        response = None
        reservation = None
        input_messages = request.get("input", [])
        if self._rate_controller is not None:
            reservation = self._rate_controller.acquire(
                estimated_tokens=estimate_chat_request_tokens(
                    messages=input_messages if isinstance(input_messages, list) else [],
                    max_tokens=self._request_max_tokens(request),
                )
            )
        try:
            response = self._client.responses.create(**request)
        except Exception as exc:
            if _is_auth_exception(exc) and self._api_key_provider is not None:
                self._ensure_api_client(force=True)
                response = self._client.responses.create(**request)
                exc = None
            if exc is not None:
                if self._rate_controller is not None and _is_rate_limit_exception(exc):
                    self._rate_controller.note_rate_limit(
                        retry_after_seconds=_retry_after_seconds_from_exception(exc)
                    )
                raise
        finally:
            if reservation is not None:
                actual_tokens = _response_total_tokens(response) if response is not None else None
                reservation.complete(actual_tokens=actual_tokens)

        usage = getattr(response, "usage", None)
        raw_message = {
            "provider": "openai_responses",
            "model": getattr(response, "model", None),
            "response_id": getattr(response, "id", None),
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
            "response": response.model_dump() if hasattr(response, "model_dump") else {"output_text": getattr(response, "output_text", "")},
        }
        visible, thinking = _extract_visible_and_thinking_from_response(
            response,
            strip_tags=self.config.thinking.strip_tags,
        )
        if not visible:
            raise _EmptyVisibleResponseError(
                thinking=thinking,
                raw_message=raw_message,
            )
        return GenerationResult(text=visible, thinking=thinking, raw_message=raw_message)

    def _generate_vertex_once(self, *, request: dict[str, Any]) -> GenerationResult:
        if self._vertex_http is None:
            raise RuntimeError("Vertex Gemini client is not initialized")
        response = None
        reservation = None
        payload = request.get("json", {})
        contents = payload.get("contents", [])
        if self._rate_controller is not None:
            reservation = self._rate_controller.acquire(
                estimated_tokens=estimate_chat_request_tokens(
                    messages=self._vertex_contents_to_messages(contents),
                    max_tokens=self._request_max_tokens(payload.get("generationConfig", {})),
                )
            )
        headers = {
            "Authorization": f"Bearer {self._resolve_api_key()}",
            "Content-Type": "application/json",
        }
        try:
            response = self._vertex_http.post(
                request["url"],
                headers=headers,
                json=payload,
            )
            if response.status_code in {401, 403} and self._api_key_provider is not None:
                headers["Authorization"] = f"Bearer {self._resolve_api_key(force_refresh=True)}"
                response = self._vertex_http.post(
                    request["url"],
                    headers=headers,
                    json=payload,
                )
            response.raise_for_status()
            response_json = response.json()
        except Exception as exc:
            if self._rate_controller is not None and _is_rate_limit_exception(exc):
                self._rate_controller.note_rate_limit(
                    retry_after_seconds=_retry_after_seconds_from_exception(exc)
                )
            raise
        finally:
            if reservation is not None:
                actual_tokens = None
                if response is not None:
                    try:
                        actual_tokens = _vertex_usage_total_tokens(response)
                    except Exception:
                        actual_tokens = None
                reservation.complete(actual_tokens=actual_tokens)

        visible, thinking, raw_message, _ = extract_vertex_generation_result(
            response_json=response_json,
            model_name=self.config.model_name,
        )
        if not visible:
            raise _EmptyVisibleResponseError(
                thinking=thinking,
                raw_message=raw_message,
            )
        return GenerationResult(text=visible, thinking=thinking, raw_message=raw_message)

    def _recover_empty_api_response(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
        vertex_response_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        wants_json_object = self._wants_json_object(response_format)
        attempts: list[tuple[list[dict[str, str]], float]] = []
        attempts.append((messages, temperature))
        if wants_json_object:
            attempts.append((self._json_repair_messages(messages, invalid_output=""), 0.0))

        if self._source == "vertex":
            last_error: Exception | None = None
            for attempt_messages, attempt_temperature in attempts:
                try:
                    recovered = self._generate_vertex_once(
                        request=self._build_vertex_request(
                            messages=attempt_messages,
                            temperature=attempt_temperature,
                            max_tokens=max(max_tokens, 512),
                            response_format={"type": "json_object"} if wants_json_object else None,
                            thinking_enabled=False,
                            vertex_response_schema=vertex_response_schema,
                        )
                    )
                    if not wants_json_object:
                        return recovered
                    return self._normalize_api_json_result(
                        messages=messages,
                        result=recovered,
                        max_tokens=max_tokens,
                        vertex_response_schema=vertex_response_schema,
                    )
                except Exception as exc:
                    last_error = exc
            if last_error is None:
                raise RuntimeError("Model response recovery failed")
            raise last_error

        last_error: Exception | None = None
        runners: list[tuple[Any, Any]] = [
            (self._build_live_api_request, self._generate_api_transport_once),
        ]
        if self._uses_responses_api():
            runners.append((self._build_api_request, self._generate_api_once))
        elif hasattr(self._client, "responses"):
            runners.append((self._build_responses_request, self._generate_responses_api_once))

        for build_request, generate_once in runners:
            for attempt_messages, attempt_temperature in attempts:
                try:
                    recovered = generate_once(
                        request=build_request(
                            messages=attempt_messages,
                            temperature=attempt_temperature,
                            max_tokens=max(max_tokens, 512),
                            response_format=None,
                            thinking_enabled=False,
                        )
                    )
                    if not wants_json_object:
                        return recovered
                    return self._normalize_api_json_result(
                        messages=messages,
                        result=recovered,
                        max_tokens=max_tokens,
                    )
                except Exception as exc:
                    last_error = exc

        if last_error is None:
            raise RuntimeError("Model response recovery failed")
        raise last_error

    def _normalize_api_json_result(
        self,
        *,
        messages: list[dict[str, Any]],
        result: GenerationResult,
        max_tokens: int,
        vertex_response_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        try:
            parsed = extract_json_object(result.text)
        except ValueError:
            repaired = self._repair_api_json_result(
                messages=messages,
                invalid_output=result.text,
                max_tokens=max_tokens,
                vertex_response_schema=vertex_response_schema,
            )
            return GenerationResult(
                text=repaired.text,
                thinking=_join_text_parts([result.thinking, repaired.thinking]),
                raw_message={
                    "initial_message": result.raw_message,
                    "repair_message": repaired.raw_message,
                },
            )
        return GenerationResult(
            text=json.dumps(parsed, ensure_ascii=False),
            thinking=result.thinking,
            raw_message=result.raw_message,
        )

    def _repair_api_json_result(
        self,
        *,
        messages: list[dict[str, Any]],
        invalid_output: str,
        max_tokens: int,
        vertex_response_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        repair_messages = self._json_repair_messages(messages, invalid_output=invalid_output)
        if self._source == "vertex":
            repair_request = self._build_vertex_request(
                messages=repair_messages,
                temperature=0.0,
                max_tokens=max(max_tokens, 512),
                response_format={"type": "json_object"},
                thinking_enabled=False,
                vertex_response_schema=vertex_response_schema,
            )
            repaired = self._generate_vertex_once(request=repair_request)
        else:
            repair_request = self._build_live_api_request(
                messages=repair_messages,
                temperature=0.0,
                max_tokens=max(max_tokens, 512),
                response_format=None,
                thinking_enabled=False,
            )
            repaired = self._generate_api_transport_once(request=repair_request)
        parsed = extract_json_object(repaired.text)
        return GenerationResult(
            text=json.dumps(parsed, ensure_ascii=False),
            thinking=repaired.thinking,
            raw_message=repaired.raw_message,
        )

    def _generate_vllm_text(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
        extra_body: dict[str, Any] | None,
    ) -> tuple[str, str, str]:
        raw = self._backend.generate(
            messages,
            self.config.model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_format=response_format,
            extra_body=extra_body,
        )
        visible, thinking = _split_thinking_from_text(
            raw,
            strip_tags=self.config.thinking.strip_tags,
        )
        if not visible:
            raise RuntimeError("Model response visible text is empty")

        if not self._wants_json_object(response_format):
            return raw, visible, thinking

        try:
            parsed = extract_json_object(visible)
        except ValueError:
            repaired_raw = self._backend.generate(
                self._json_repair_messages(messages, visible),
                self.config.model_name,
                temperature=0.0,
                max_output_tokens=max(max_tokens, 512),
                response_format=None,
            )
            repaired_visible, repaired_thinking = _split_thinking_from_text(
                repaired_raw,
                strip_tags=self.config.thinking.strip_tags,
            )
            parsed = extract_json_object(repaired_visible)
            canonical = json.dumps(parsed, ensure_ascii=False)
            raw_payload = json.dumps(
                {
                    "initial_content": raw,
                    "repair_content": repaired_raw,
                },
                ensure_ascii=False,
            )
            return raw_payload, canonical, _join_text_parts([thinking, repaired_thinking])

        canonical = json.dumps(parsed, ensure_ascii=False)
        return raw, canonical, thinking

    @staticmethod
    def _wants_json_object(response_format: dict[str, Any] | None) -> bool:
        if not isinstance(response_format, dict):
            return False
        return str(response_format.get("type") or "").strip().lower() == "json_object"

    @staticmethod
    def _json_repair_messages(messages: list[dict[str, Any]], invalid_output: str) -> list[dict[str, str]]:
        def _message_text(item: dict[str, Any]) -> str:
            content = item.get("content")
            if isinstance(content, str):
                return content
            parts = item.get("parts")
            if isinstance(parts, list):
                texts = [
                    str(part.get("text") or "").strip()
                    for part in parts
                    if isinstance(part, dict) and str(part.get("text") or "").strip()
                ]
                return _join_text_parts(texts)
            return ""

        schema_context = "\n\n".join(
            f"{item.get('role', 'user').upper()}:\n{_message_text(item)}"
            for item in messages
        ).strip()
        repair_instruction = (
            "You are a strict JSON formatter. "
            "Read the original prompt and the invalid assistant answer, then output exactly one valid JSON object only. "
            "Do not add markdown, code fences, commentary, or analysis. "
            "Preserve the original answer's intent while making it match the requested JSON schema."
        )
        return [
            {"role": "system", "content": repair_instruction},
            {
                "role": "user",
                "content": (
                    f"Original prompt:\n{schema_context}\n\n"
                    + (
                        f"Invalid assistant answer:\n{invalid_output}"
                        if invalid_output.strip()
                        else "The previous assistant answer was empty. Answer the original prompt now."
                    )
                ),
            },
        ]


def _is_rate_limit_exception(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429
    return isinstance(exc, APIStatusError) and exc.status_code == 429


def _is_auth_exception(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {401, 403}
    return isinstance(exc, APIStatusError) and exc.status_code in {401, 403}


def _retry_after_seconds_from_exception(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    if value is None:
        value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _response_total_tokens(response: Any) -> int | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None and isinstance(usage, dict):
        total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        return None
    try:
        normalized = int(total_tokens)
    except (TypeError, ValueError):
        return None
    return normalized if normalized > 0 else None


def _vertex_usage_total_tokens(response: httpx.Response) -> int | None:
    try:
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usageMetadata")
    if not isinstance(usage, dict):
        return None
    total_tokens = usage.get("totalTokenCount")
    if total_tokens is None:
        return None
    try:
        normalized = int(total_tokens)
    except (TypeError, ValueError):
        return None
    return normalized if normalized > 0 else None

OpenAIReasoningClient = ReasoningClient
