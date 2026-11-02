from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Protocol
import json
import os
import tempfile
import time
import uuid

from openai import OpenAI

from EmoTrack.utils.openai_compat import extract_batch_message_content, prefers_max_completion_tokens
from EmoTrack.utils.json_utils import extract_json_object


DEFAULT_OPENAI_BASE_URL = "https://api.studio.nebius.com/v1/"


class InferenceBackend(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float = 0.6,
        max_output_tokens: int = 256,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        ...

    def generate_json_object(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float = 0.6,
        max_output_tokens: int = 4096,
    ) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class VLLMRuntimeConfig:
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    kv_cache_memory_bytes: int | None = None
    max_model_len: int | None = 8192
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    enforce_eager: bool = False
    enable_chunked_prefill: bool | None = None
    enable_prefix_caching: bool | None = None
    gdn_prefill_backend: str | None = None
    dtype: str = "auto"
    seed: int | None = None
    trust_remote_code: bool = False


@dataclass(frozen=True)
class InferenceConfig:
    backend: str
    model: str
    api_key: str = ""
    base_url: str | None = DEFAULT_OPENAI_BASE_URL
    request_mode: str = "quick"
    vllm: VLLMRuntimeConfig = field(default_factory=VLLMRuntimeConfig)


def _extract_message_content(raw: Any) -> str:
    if isinstance(raw, str):
        text = raw.strip()
        if text:
            return text
        raise RuntimeError("Model response content is empty")

    choices = getattr(raw, "choices", None)
    if not choices:
        raise RuntimeError("Model response has no choices")

    message = choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        text = content.strip()
        if text:
            return text
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_part = str(item.get("text", "")).strip()
            else:
                text_part = str(getattr(item, "text", "")).strip()
            if text_part:
                parts.append(text_part)
        joined = " ".join(parts).strip()
        if joined:
            return joined

    raise RuntimeError("Model response content is empty")


@dataclass
class OpenAIAPIBackend:
    api_key: str
    base_url: str | None = DEFAULT_OPENAI_BASE_URL
    request_mode: str = "quick"
    batch_poll_seconds: int = 2

    def __post_init__(self) -> None:
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    def _build_chat_request(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float,
        max_output_tokens: int,
        response_format: dict[str, Any] | None,
        extra_body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            request["response_format"] = response_format
        payload_extra = dict(extra_body or {})
        top_p = payload_extra.pop("top_p", None)
        if top_p is not None:
            request["top_p"] = top_p
        if payload_extra:
            request["extra_body"] = payload_extra
        token_param = "max_completion_tokens" if prefers_max_completion_tokens(model) else "max_tokens"
        request[token_param] = max_output_tokens
        return request

    def _generate_via_responses(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        input_payload = [
            {"role": item["role"], "content": item["content"]}
            for item in messages
        ]

        try:
            response = self._client.responses.create(
                model=model,
                input=input_payload,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            text = getattr(response, "output_text", "") or ""
            if text.strip():
                return text.strip()
        except Exception:
            pass

        response = self._client.responses.create(
            model=model,
            input=input_payload,
            max_output_tokens=max_output_tokens,
        )
        text = getattr(response, "output_text", "") or ""
        if not text.strip():
            raise RuntimeError("Responses API returned empty output_text")
        return text.strip()

    def _generate_via_batch(self, request: dict[str, Any]) -> str:
        custom_id = f"req_{uuid.uuid4().hex[:12]}"
        batch_line = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": request,
        }

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
                temp_path = tmp.name
                tmp.write(json.dumps(batch_line, ensure_ascii=False) + "\n")

            with open(temp_path, "rb") as file_obj:
                input_file = self._client.files.create(file=file_obj, purpose="batch")

            batch_job = self._client.batches.create(
                input_file_id=input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            while True:
                batch_job = self._client.batches.retrieve(batch_job.id)
                status = str(getattr(batch_job, "status", "")).lower()
                if status in {"completed", "done"}:
                    break
                if status in {"failed", "canceled", "cancelled", "error"}:
                    raise RuntimeError(f"Batch request failed with status={status} (batch_id={batch_job.id})")
                time.sleep(self.batch_poll_seconds)

            completed = self._client.batches.retrieve(batch_job.id)
            output_file_id = getattr(completed, "output_file_id", None)
            if not output_file_id:
                raise RuntimeError(f"Batch finished without output_file_id (batch_id={batch_job.id})")

            output_content = self._client.files.content(output_file_id)
            raw_text = output_content.text if hasattr(output_content, "text") else output_content.read().decode("utf-8")
            for line in raw_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("custom_id") != custom_id:
                    continue
                content = extract_batch_message_content(row)
                if content:
                    return content
            raise RuntimeError(f"Batch output missing response content (batch_id={batch_job.id})")
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float = 0.6,
        max_output_tokens: int = 256,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        request = self._build_chat_request(
            messages,
            model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            extra_body=extra_body,
        )

        if self.request_mode == "batch":
            return self._generate_via_batch(request)

        if response_format is None and not extra_body:
            try:
                return self._generate_via_responses(
                    messages,
                    model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            except Exception:
                pass

        response = self._client.chat.completions.create(**request)
        return _extract_message_content(response)

    def generate_json_object(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float = 0.6,
        max_output_tokens: int = 4096,
    ) -> dict[str, Any]:
        raw = self.generate(
            messages,
            model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format={"type": "json_object"},
        )
        return extract_json_object(raw)

    def close(self) -> None:
        return None


@dataclass
class LocalVLLMBackend:
    model: str
    runtime: VLLMRuntimeConfig

    def __post_init__(self) -> None:
        try:
            vllm_module = import_module("vllm")
        except ImportError as exc:
            raise RuntimeError(
                "vLLM inference backend was selected but the 'vllm' package is not installed."
            ) from exc

        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "tensor_parallel_size": self.runtime.tensor_parallel_size,
            "gpu_memory_utilization": self.runtime.gpu_memory_utilization,
            "enforce_eager": self.runtime.enforce_eager,
            "dtype": self.runtime.dtype,
            "trust_remote_code": self.runtime.trust_remote_code,
        }
        if self.runtime.kv_cache_memory_bytes is not None:
            llm_kwargs["kv_cache_memory_bytes"] = self.runtime.kv_cache_memory_bytes
        if self.runtime.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.runtime.max_model_len
        if self.runtime.max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = self.runtime.max_num_batched_tokens
        if self.runtime.max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = self.runtime.max_num_seqs
        if self.runtime.enable_chunked_prefill is not None:
            llm_kwargs["enable_chunked_prefill"] = self.runtime.enable_chunked_prefill
        if self.runtime.enable_prefix_caching is not None:
            llm_kwargs["enable_prefix_caching"] = self.runtime.enable_prefix_caching
        if self.runtime.seed is not None:
            llm_kwargs["seed"] = self.runtime.seed

        self._SamplingParams = getattr(vllm_module, "SamplingParams")
        self._llm = getattr(vllm_module, "LLM")(**llm_kwargs)
        self._tokenizer = self._llm.get_tokenizer()

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        return self._render_prompt_with_extras(messages, extra_body=None)

    def _render_prompt_with_extras(
        self,
        messages: list[dict[str, str]],
        *,
        extra_body: dict[str, Any] | None,
    ) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            chat_template_kwargs = None
            if isinstance(extra_body, dict):
                raw_chat_template_kwargs = extra_body.get("chat_template_kwargs")
                if isinstance(raw_chat_template_kwargs, dict):
                    chat_template_kwargs = raw_chat_template_kwargs
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template_kwargs=chat_template_kwargs,
                )
            except TypeError:
                try:
                    return self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except TypeError:
                    return self._tokenizer.apply_chat_template(messages, tokenize=False)

        lines = [f"{item['role'].title()}: {item['content']}" for item in messages]
        lines.append("Assistant:")
        return "\n\n".join(lines)

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float = 0.6,
        max_output_tokens: int = 256,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        _ = response_format
        if model and model != self.model:
            raise ValueError(f"Local vLLM backend loaded model '{self.model}', got request for '{model}'")

        prompt = self._render_prompt_with_extras(messages, extra_body=extra_body)
        sampling_kwargs: dict[str, Any] = {
            "temperature": max(0.0, temperature),
            "max_tokens": max_output_tokens,
        }
        if isinstance(extra_body, dict):
            top_p = extra_body.get("top_p")
            top_k = extra_body.get("top_k")
            repetition_penalty = extra_body.get("repetition_penalty")
            frequency_penalty = extra_body.get("frequency_penalty")
            if top_p is not None:
                sampling_kwargs["top_p"] = float(top_p)
            if top_k is not None:
                sampling_kwargs["top_k"] = int(top_k)
            if repetition_penalty is not None:
                sampling_kwargs["repetition_penalty"] = float(repetition_penalty)
            if frequency_penalty is not None:
                sampling_kwargs["frequency_penalty"] = float(frequency_penalty)
        sampling_params = self._SamplingParams(**sampling_kwargs)
        try:
            outputs = self._llm.generate([prompt], sampling_params, use_tqdm=False)
        except TypeError:
            outputs = self._llm.generate([prompt], sampling_params)

        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no candidate outputs")
        text = str(outputs[0].outputs[0].text).strip()
        if not text:
            raise RuntimeError("vLLM returned empty text")
        return text

    def generate_json_object(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float = 0.6,
        max_output_tokens: int = 4096,
    ) -> dict[str, Any]:
        raw = self.generate(
            messages,
            model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format={"type": "json_object"},
        )
        return extract_json_object(raw)

    def close(self) -> None:
        return None


def build_inference_backend(config: InferenceConfig) -> InferenceBackend:
    backend_name = config.backend.strip().lower()
    model = config.model.strip()
    if not model:
        raise ValueError("Inference model must be provided")

    if backend_name == "openai":
        api_key = config.api_key.strip()
        if not api_key:
            raise ValueError("OpenAI-compatible inference backend requires a non-empty API key")
        return OpenAIAPIBackend(
            api_key=api_key,
            base_url=(config.base_url or "").strip() or None,
            request_mode=config.request_mode.strip().lower(),
        )

    if backend_name == "vllm":
        if config.request_mode.strip().lower() == "batch":
            raise ValueError("Local vLLM backend does not support batch request mode")
        return LocalVLLMBackend(model=model, runtime=config.vllm)

    raise ValueError(f"Unsupported inference backend: {config.backend}")
