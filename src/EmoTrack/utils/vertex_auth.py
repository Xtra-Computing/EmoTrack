from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib
import os
import threading
import time
from typing import Any
from urllib.parse import urlparse


_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def is_vertex_base_url(base_url: str | None) -> bool:
    host = (urlparse(base_url or "").hostname or "").strip().lower()
    return host.endswith("aiplatform.googleapis.com")


def is_vertex_openai_base_url(base_url: str | None) -> bool:
    if not is_vertex_base_url(base_url):
        return False
    path = (urlparse(base_url or "").path or "").strip().rstrip("/")
    return path.endswith("/endpoints/openapi")


def normalize_vertex_api_root(base_url: str | None) -> str:
    raw = str(base_url or "").strip().rstrip("/")
    if not raw:
        raise ValueError("Vertex AI base URL is empty")
    if not is_vertex_base_url(raw):
        raise ValueError(f"Not a Vertex AI base URL: {raw}")
    if is_vertex_openai_base_url(raw):
        raise ValueError(
            "models.source=vertex expects the native Vertex Gemini API root, not the "
            "OpenAI-compatible '/endpoints/openapi' endpoint."
        )
    path = (urlparse(raw).path or "").strip()
    if "/v1/projects/" not in path or "/locations/" not in path:
        raise ValueError(
            "Vertex Gemini base_url must look like "
            "'https://aiplatform.googleapis.com/v1/projects/<project>/locations/<location>'."
        )
    return raw


def normalize_vertex_model_name(model_name: str) -> str:
    normalized = str(model_name or "").strip().strip("/")
    if not normalized:
        raise ValueError("Vertex Gemini model_name is empty")
    if normalized.startswith("google/"):
        normalized = normalized.split("/", 1)[1].strip()
    return normalized


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _load_google_auth_modules() -> tuple[Any, Any]:
    try:
        auth_module = importlib.import_module("google.auth")
        requests_module = importlib.import_module("google.auth.transport.requests")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Vertex AI OpenAI-compatible calls require ADC-backed token refresh via "
            "'google-auth[requests]'. Install project dependencies and run "
            "'gcloud auth application-default login'."
        ) from exc
    return auth_module, requests_module


def _expiry_to_monotonic(expiry: Any, *, refresh_margin_seconds: int) -> float:
    if isinstance(expiry, datetime):
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        seconds_until_expiry = max(0.0, expiry.timestamp() - time.time())
        return time.monotonic() + max(0.0, seconds_until_expiry - refresh_margin_seconds)
    return time.monotonic() + max(0, 3600 - refresh_margin_seconds)


@dataclass
class VertexAccessTokenProvider:
    base_url: str | None
    initial_api_key: str = ""
    refresh_margin_seconds: int = field(
        default_factory=lambda: _env_int("EMOTRACK_VERTEX_REFRESH_MARGIN_SECONDS", 300)
    )

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _token: str = field(default="", init=False, repr=False)
    _refresh_at_monotonic: float = field(default=0.0, init=False, repr=False)
    _credentials: Any = field(default=None, init=False, repr=False)
    _request: Any = field(default=None, init=False, repr=False)

    @property
    def enabled(self) -> bool:
        return is_vertex_base_url(self.base_url)

    def _ensure_adc(self) -> None:
        if self._credentials is not None and self._request is not None:
            return
        auth_module, requests_module = _load_google_auth_modules()
        try:
            credentials, _ = auth_module.default(scopes=[_CLOUD_PLATFORM_SCOPE])
        except Exception as exc:
            raise RuntimeError(
                "Application Default Credentials are required for Vertex AI "
                "auth. Run 'gcloud auth application-default login' first."
            ) from exc
        self._credentials = credentials
        self._request = requests_module.Request()

    def current_api_key(self) -> str:
        if not self.enabled:
            return str(self.initial_api_key or "").strip()
        now = time.monotonic()
        if self._token and now < self._refresh_at_monotonic:
            return self._token
        return self.force_refresh()

    def force_refresh(self) -> str:
        if not self.enabled:
            return str(self.initial_api_key or "").strip()
        with self._lock:
            now = time.monotonic()
            if self._token and now < self._refresh_at_monotonic:
                return self._token
            self._ensure_adc()
            self._credentials.refresh(self._request)
            token = str(getattr(self._credentials, "token", "") or "").strip()
            if not token:
                raise RuntimeError("ADC refresh completed without returning an access token")
            self._token = token
            self._refresh_at_monotonic = _expiry_to_monotonic(
                getattr(self._credentials, "expiry", None),
                refresh_margin_seconds=self.refresh_margin_seconds,
            )
            return self._token
