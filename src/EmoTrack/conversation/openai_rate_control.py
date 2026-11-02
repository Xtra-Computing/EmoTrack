from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import os
import threading
import time


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class _TokenReservation:
    reservation_id: int
    created_at: float
    reserved_tokens: int


@dataclass(frozen=True)
class TokenRateReservation:
    controller: "_OpenAITokenRateController"
    reservation_id: int

    def complete(self, *, actual_tokens: int | None = None) -> None:
        self.controller.complete(self.reservation_id, actual_tokens=actual_tokens)


class _OpenAITokenRateController:
    def __init__(self, *, endpoint_key: str) -> None:
        self.endpoint_key = endpoint_key
        self.window_seconds = max(1.0, _env_float("EMOTRACK_OPENAI_RATE_WINDOW_SECONDS", 60.0))
        self.base_tpm_limit = max(0, _env_int("EMOTRACK_OPENAI_TPM_LIMIT", 0))
        self.base_rpm_limit = max(0, _env_int("EMOTRACK_OPENAI_RPM_LIMIT", 0))
        self.max_inflight = max(0, _env_int("EMOTRACK_OPENAI_MAX_INFLIGHT", 0))
        self.reduction_factor = min(0.99, max(0.10, _env_float("EMOTRACK_OPENAI_TPM_REDUCTION_FACTOR", 0.80)))
        self.recovery_factor = max(1.0, _env_float("EMOTRACK_OPENAI_TPM_RECOVERY_FACTOR", 1.10))
        self.recovery_seconds = max(5.0, _env_float("EMOTRACK_OPENAI_TPM_RECOVERY_SECONDS", 300.0))
        self.default_cooldown_seconds = max(1.0, _env_float("EMOTRACK_OPENAI_RATE_LIMIT_COOLDOWN_SECONDS", 20.0))
        self._adaptive_tpm_limit = self.base_tpm_limit
        self._min_tpm_limit = max(1000, int(self.base_tpm_limit * 0.20)) if self.base_tpm_limit > 0 else 0

        self._recent: deque[_TokenReservation] = deque()
        self._active: set[int] = set()
        self._next_reservation_id = 1
        self._cooldown_until = 0.0
        self._last_rate_limit_at = 0.0
        self._last_recovery_at = 0.0
        self._last_wait_log_at = 0.0
        self._last_limit_log_at = 0.0
        self._condition = threading.Condition()

    @property
    def enabled(self) -> bool:
        return self.base_tpm_limit > 0 or self.base_rpm_limit > 0 or self.max_inflight > 0

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._recent and self._recent[0].created_at <= cutoff:
            self._recent.popleft()

    def _recover_if_needed(self, now: float) -> None:
        if self.base_tpm_limit <= 0 or self._adaptive_tpm_limit <= 0:
            return
        if self._adaptive_tpm_limit >= self.base_tpm_limit:
            return
        if self._last_rate_limit_at <= 0:
            return
        if now - self._last_rate_limit_at < self.recovery_seconds:
            return
        if self._last_recovery_at > 0 and now - self._last_recovery_at < self.recovery_seconds:
            return
        recovered = min(self.base_tpm_limit, int(math.ceil(self._adaptive_tpm_limit * self.recovery_factor)))
        if recovered != self._adaptive_tpm_limit:
            self._adaptive_tpm_limit = recovered
            self._last_recovery_at = now
            print(
                f"[openai-rate] endpoint={self.endpoint_key} recovered_tpm_limit={self._adaptive_tpm_limit}",
                flush=True,
            )

    def _tpm_limit(self) -> int:
        return max(0, self._adaptive_tpm_limit)

    def _token_usage(self) -> int:
        return sum(item.reserved_tokens for item in self._recent)

    def _wait_for_token_capacity(self, *, now: float, requested_tokens: int, limit: int) -> float:
        if limit <= 0:
            return 0.0
        remaining = max(0, self._token_usage() + requested_tokens - limit)
        if remaining <= 0:
            return 0.0

        freed = 0
        for item in self._recent:
            freed += item.reserved_tokens
            if freed >= remaining:
                return max(0.05, item.created_at + self.window_seconds - now)
        return max(0.05, self.window_seconds)

    def _wait_for_request_capacity(self, *, now: float) -> float:
        if self.base_rpm_limit <= 0 or len(self._recent) < self.base_rpm_limit:
            return 0.0
        item = self._recent[0]
        return max(0.05, item.created_at + self.window_seconds - now)

    def acquire(self, *, estimated_tokens: int) -> TokenRateReservation | None:
        if not self.enabled:
            return None

        with self._condition:
            while True:
                now = time.monotonic()
                self._prune(now)
                self._recover_if_needed(now)

                wait_seconds = 0.0
                if self._cooldown_until > now:
                    wait_seconds = self._cooldown_until - now
                elif self.max_inflight > 0 and len(self._active) >= self.max_inflight:
                    wait_seconds = 0.10
                else:
                    tpm_limit = self._tpm_limit()
                    reserved_tokens = max(1, estimated_tokens)
                    if tpm_limit > 0:
                        reserved_tokens = min(reserved_tokens, tpm_limit)
                    wait_seconds = max(
                        self._wait_for_token_capacity(now=now, requested_tokens=reserved_tokens, limit=tpm_limit),
                        self._wait_for_request_capacity(now=now),
                    )
                    if wait_seconds <= 0.0:
                        reservation_id = self._next_reservation_id
                        self._next_reservation_id += 1
                        self._recent.append(
                            _TokenReservation(
                                reservation_id=reservation_id,
                                created_at=now,
                                reserved_tokens=reserved_tokens,
                            )
                        )
                        self._active.add(reservation_id)
                        return TokenRateReservation(controller=self, reservation_id=reservation_id)

                if now - self._last_wait_log_at >= 5.0 and wait_seconds >= 1.0:
                    self._last_wait_log_at = now
                    print(
                        "[openai-rate] waiting "
                        f"endpoint={self.endpoint_key} wait_seconds={wait_seconds:.1f} "
                        f"inflight={len(self._active)} token_usage={self._token_usage()} "
                        f"tpm_limit={self._tpm_limit()} rpm_limit={self.base_rpm_limit}",
                        flush=True,
                    )
                self._condition.wait(timeout=max(0.05, wait_seconds))

    def complete(self, reservation_id: int, *, actual_tokens: int | None = None) -> None:
        if not self.enabled:
            return

        normalized_tokens = None
        if actual_tokens is not None and actual_tokens > 0:
            normalized_tokens = int(actual_tokens)

        with self._condition:
            if reservation_id in self._active:
                self._active.remove(reservation_id)
            if normalized_tokens is not None:
                for item in self._recent:
                    if item.reservation_id == reservation_id:
                        item.reserved_tokens = max(1, normalized_tokens)
                        break
            self._condition.notify_all()

    def note_rate_limit(self, *, retry_after_seconds: float | None = None) -> None:
        if not self.enabled:
            return

        with self._condition:
            now = time.monotonic()
            cooldown = retry_after_seconds if retry_after_seconds and retry_after_seconds > 0 else self.default_cooldown_seconds
            self._cooldown_until = max(self._cooldown_until, now + cooldown)
            self._last_rate_limit_at = now
            if self.base_tpm_limit > 0:
                reduced = max(self._min_tpm_limit, int(math.floor(self._adaptive_tpm_limit * self.reduction_factor)))
                self._adaptive_tpm_limit = max(1, reduced)
            if now - self._last_limit_log_at >= 1.0:
                self._last_limit_log_at = now
                print(
                    "[openai-rate] rate_limit "
                    f"endpoint={self.endpoint_key} cooldown_seconds={cooldown:.1f} "
                    f"adaptive_tpm_limit={self._adaptive_tpm_limit}",
                    flush=True,
                )
            self._condition.notify_all()


_CONTROLLER_LOCK = threading.Lock()
_CONTROLLERS: dict[str, _OpenAITokenRateController] = {}


def estimate_chat_request_tokens(*, messages: list[dict[str, str]], max_tokens: int) -> int:
    prompt_chars = 0
    for message in messages:
        prompt_chars += len(str(message.get("role", "") or ""))
        prompt_chars += len(str(message.get("content", "") or ""))
    prompt_tokens = int(math.ceil(prompt_chars / 4.0))
    return max(1, prompt_tokens + max(0, int(max_tokens)) + 32)


def get_openai_chat_rate_controller(*, base_url: str, model_name: str) -> _OpenAITokenRateController | None:
    endpoint_key = f"{str(base_url or '').rstrip('/')}::{str(model_name or '').strip()}"
    with _CONTROLLER_LOCK:
        controller = _CONTROLLERS.get(endpoint_key)
        if controller is None:
            controller = _OpenAITokenRateController(endpoint_key=endpoint_key)
            _CONTROLLERS[endpoint_key] = controller
    return controller if controller.enabled else None
