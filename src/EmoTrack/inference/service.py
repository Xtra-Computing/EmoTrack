from __future__ import annotations

import atexit
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from EmoTrack.utils.paths import REPO_ROOT


DEFAULT_LOCAL_SERVICE_LOG_ROOT = REPO_ROOT / "out" / "local_services"
DEFAULT_VLLM_SERVER_ARGS: dict[str, Any] = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
    "kv_cache_memory_bytes": None,
    "max_model_len": 8192,
    "max_num_batched_tokens": 2048,
    "max_num_seqs": 256,
    "dtype": "bfloat16",
    "seed": 0,
    "trust_remote_code": False,
    "enforce_eager": True,
    "enable_chunked_prefill": True,
    "enable_prefix_caching": False,
}
VLLM_VALUE_FLAGS: tuple[tuple[str, str], ...] = (
    ("tensor_parallel_size", "--tensor-parallel-size"),
    ("gpu_memory_utilization", "--gpu-memory-utilization"),
    ("kv_cache_memory_bytes", "--kv-cache-memory-bytes"),
    ("max_model_len", "--max-model-len"),
    ("max_num_batched_tokens", "--max-num-batched-tokens"),
    ("max_num_seqs", "--max-num-seqs"),
    ("dtype", "--dtype"),
    ("seed", "--seed"),
    ("gdn_prefill_backend", "--gdn-prefill-backend"),
)
VLLM_BOOL_FLAGS: tuple[tuple[str, str], ...] = (
    ("trust_remote_code", "--trust-remote-code"),
    ("enforce_eager", "--enforce-eager"),
    ("enable_chunked_prefill", "--enable-chunked-prefill"),
    ("enable_prefix_caching", "--enable-prefix-caching"),
)

_SPAWNED_VLLM_SERVERS: list[subprocess.Popen[bytes]] = []
_PROCESS_LOG_PATHS: dict[int, Path] = {}
_SIGNAL_CLEANUP_INSTALLED = False
_PRESERVE_MANAGED_VLLM_SERVERS = False


def resolve_vllm_base_url(raw_url: str) -> str:
    parsed = urlparse(raw_url.strip().rstrip("/"))
    path = parsed.path
    for suffix in ("/rerank", "/embeddings", "/completions", "/chat/completions"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def base_url_looks_local(raw_url: str) -> bool:
    host = (urlparse(raw_url).hostname or "").strip().lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def auth_headers(api_key: str) -> dict[str, str]:
    key = str(api_key or "").strip()
    if not key or key.upper() == "EMPTY":
        return {}
    return {"Authorization": f"Bearer {key}"}


def check_endpoint_reachable(base_url: str, label: str) -> None:
    check_endpoint_reachable_with_api_key(base_url, label, api_key="")


def check_endpoint_reachable_with_api_key(base_url: str, label: str, *, api_key: str) -> None:
    normalized = resolve_vllm_base_url(base_url)
    if not normalized:
        raise RuntimeError(f"{label} endpoint is empty in config")
    headers = auth_headers(api_key)
    try:
        response = httpx.get(f"{normalized}/models", timeout=5.0, headers=headers or None)
        if response.status_code == 401 and not headers:
            raise RuntimeError(
                f"{label} endpoint {normalized} requires authentication; "
                "configure an API key in config or environment."
            )
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"{label} endpoint {normalized} is not reachable: {exc}") from exc


def verify_model_served(base_url: str, model_name: str, label: str) -> None:
    verify_model_served_with_api_key(base_url, model_name, label, api_key="")


def verify_model_served_with_api_key(base_url: str, model_name: str, label: str, *, api_key: str) -> None:
    normalized = resolve_vllm_base_url(base_url)
    if not normalized or not model_name.strip():
        return
    headers = auth_headers(api_key)
    try:
        response = httpx.get(f"{normalized}/models", timeout=5.0, headers=headers or None)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    data = payload.get("data", [])
    if not isinstance(data, list):
        return
    served_ids = [
        str(item.get("id") or "").strip()
        for item in data
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    ]
    if not served_ids:
        return
    if model_name not in served_ids:
        raise RuntimeError(
            f"{label} model '{model_name}' is not served by {normalized}. "
            f"Available: {', '.join(served_ids)}. "
            f"Fix the model name in your config or start the correct vLLM process."
        )


def port_from_url(url: str) -> int:
    parsed = urlparse(url)
    if parsed.port is not None:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80


def replace_url_port(url: str, port: int) -> str:
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not parsed.scheme or not parsed.netloc or not hostname:
        raise ValueError(f"Cannot derive worker URL from base URL: {url}")

    host = hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    auth = ""
    if parsed.username:
        auth = parsed.username
        if parsed.password is not None:
            auth += f":{parsed.password}"
        auth += "@"

    netloc = f"{auth}{host}:{port}"
    return parsed._replace(netloc=netloc).geturl()


def split_visible_gpus(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def validate_disjoint_visible_gpu_workers(worker_visible_gpus: list[str]) -> None:
    seen_gpu_to_worker: dict[str, int] = {}
    for worker_index, raw_visible_gpus in enumerate(worker_visible_gpus, start=1):
        gpu_ids = split_visible_gpus(raw_visible_gpus)
        if not gpu_ids:
            raise ValueError("Each llm.worker_visible_gpus entry must resolve to at least one CUDA device")
        for gpu_id in gpu_ids:
            previous_worker = seen_gpu_to_worker.get(gpu_id)
            if previous_worker is None:
                seen_gpu_to_worker[gpu_id] = worker_index
                continue
            raise ValueError(
                f"Evaluation workers must use disjoint GPUs; "
                f"GPU {gpu_id} was assigned to worker {previous_worker} and worker {worker_index}"
            )


def validate_tensor_parallel_size_for_workers(
    *,
    tensor_parallel_size: int,
    worker_visible_gpus: list[str],
) -> None:
    if tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be >= 1")
    for worker_index, raw_visible_gpus in enumerate(worker_visible_gpus, start=1):
        gpu_count = len(split_visible_gpus(raw_visible_gpus))
        if tensor_parallel_size > gpu_count:
            raise ValueError(
                f"tensor_parallel_size={tensor_parallel_size} requires at least {tensor_parallel_size} GPUs "
                f"per evaluation worker, but worker {worker_index} only has {gpu_count}: {raw_visible_gpus}"
            )


def allocate_worker_base_urls(base_url: str, num_workers: int) -> list[str]:
    if num_workers <= 0:
        return []
    urls = [resolve_vllm_base_url(base_url)]
    used_ports = {port_from_url(urls[0])}
    next_port = port_from_url(urls[0]) + 1
    while len(urls) < num_workers:
        while next_port in used_ports:
            next_port += 1
        urls.append(replace_url_port(urls[0], next_port))
        used_ports.add(next_port)
        next_port += 1
    return urls


def is_endpoint_up(base_url: str) -> bool:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/models", timeout=3.0)
        return response.status_code == 200
    except Exception:
        return False


def service_log_path(*, kind: str, label: str, port: int, root: Path | None = None) -> Path:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._") or kind
    log_root = (root or DEFAULT_LOCAL_SERVICE_LOG_ROOT).expanduser().resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    return log_root / f"{kind}_{safe_label}_{port}.log"


def vllm_executable() -> str:
    repo_local = REPO_ROOT / ".venv" / "bin" / "vllm"
    if repo_local.exists():
        return str(repo_local)
    resolved = shutil.which("vllm")
    if resolved:
        return resolved
    raise FileNotFoundError(
        f"vllm executable not found; expected repo-local {repo_local} or PATH entry"
    )


def process_stderr_tail(proc: subprocess.Popen[bytes], max_chars: int = 4000) -> str:
    log_path = _PROCESS_LOG_PATHS.get(proc.pid or -1)
    if log_path is not None and log_path.exists():
        try:
            payload = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            payload = ""
    elif proc.stderr is not None:
        try:
            payload = proc.stderr.read().decode("utf-8", errors="replace")
        except Exception:
            payload = ""
    else:
        return ""
    payload = payload.strip()
    if len(payload) <= max_chars:
        return payload
    return payload[-max_chars:]


def process_log_path(proc: subprocess.Popen[bytes]) -> Path | None:
    return _PROCESS_LOG_PATHS.get(proc.pid or -1)


def managed_vllm_log_paths() -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for path in _PROCESS_LOG_PATHS.values():
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def build_vllm_server_args(vllm_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "tensor_parallel_size": int(
            vllm_cfg.get("tensor_parallel_size", DEFAULT_VLLM_SERVER_ARGS["tensor_parallel_size"])
        ),
        "gpu_memory_utilization": float(
            vllm_cfg.get("gpu_memory_utilization", DEFAULT_VLLM_SERVER_ARGS["gpu_memory_utilization"])
        ),
        "kv_cache_memory_bytes": _optional_int(
            vllm_cfg.get("kv_cache_memory_bytes", DEFAULT_VLLM_SERVER_ARGS["kv_cache_memory_bytes"])
        ),
        "max_model_len": int(
            vllm_cfg.get("max_model_len", DEFAULT_VLLM_SERVER_ARGS["max_model_len"])
        ),
        "max_num_batched_tokens": int(
            vllm_cfg.get("max_num_batched_tokens", DEFAULT_VLLM_SERVER_ARGS["max_num_batched_tokens"])
        ),
        "max_num_seqs": int(
            vllm_cfg.get("max_num_seqs", DEFAULT_VLLM_SERVER_ARGS["max_num_seqs"])
        ),
        "dtype": str(vllm_cfg.get("dtype", DEFAULT_VLLM_SERVER_ARGS["dtype"])),
        "seed": int(vllm_cfg.get("seed", DEFAULT_VLLM_SERVER_ARGS["seed"])),
        "trust_remote_code": bool(
            vllm_cfg.get("trust_remote_code", DEFAULT_VLLM_SERVER_ARGS["trust_remote_code"])
        ),
        "enforce_eager": bool(
            vllm_cfg.get("enforce_eager", DEFAULT_VLLM_SERVER_ARGS["enforce_eager"])
        ),
        "enable_chunked_prefill": bool(
            vllm_cfg.get("enable_chunked_prefill", DEFAULT_VLLM_SERVER_ARGS["enable_chunked_prefill"])
        ),
        "enable_prefix_caching": bool(
            vllm_cfg.get("enable_prefix_caching", DEFAULT_VLLM_SERVER_ARGS["enable_prefix_caching"])
        ),
        "gdn_prefill_backend": str(vllm_cfg.get("gdn_prefill_backend", "")) or None,
    }


def wait_for_endpoints(
    pending: list[tuple[str, str, subprocess.Popen[bytes]]],
    *,
    timeout_sec: int = 600,
    log_prefix: str = "[llm]",
) -> None:
    start = time.time()
    last_msg = 0
    remaining = {i for i in range(len(pending))}
    while time.time() - start < timeout_sec:
        for index in list(remaining):
            base_url, label, proc = pending[index]
            if is_endpoint_up(base_url):
                print(f"{log_prefix} {label} server ready at {base_url}", flush=True)
                remaining.discard(index)
                continue
            if proc.poll() is not None:
                stderr_tail = process_stderr_tail(proc)
                details = f" exited with code {proc.returncode}"
                if stderr_tail:
                    details += f"\n{stderr_tail}"
                raise RuntimeError(f"{label} server failed to start:{details}")
        if not remaining:
            return
        elapsed = int(time.time() - start)
        if elapsed - last_msg >= 10:
            labels = [pending[index][1] for index in remaining]
            print(f"{log_prefix} waiting for {', '.join(labels)}... ({elapsed}s)", flush=True)
            last_msg = elapsed
        time.sleep(2)
    still_pending = [pending[index][1] for index in remaining]
    raise RuntimeError(f"Servers {still_pending} did not start within {timeout_sec}s")


def spawn_vllm_server(
    model: str,
    port: int,
    base_url: str,
    visible_gpus: str,
    label: str,
    *,
    runner: str | None = None,
    convert: str | None = None,
    vllm_args: dict[str, Any] | None = None,
    reasoning_parser: str | None = None,
    log_root: Path | None = None,
    log_prefix: str = "[llm]",
) -> tuple[str, str, subprocess.Popen[bytes]] | None:
    if is_endpoint_up(base_url):
        print(f"{log_prefix} {label} already running at {base_url}", flush=True)
        return None

    cmd = [vllm_executable(), "serve", model, "--port", str(port),
           "--default-chat-template-kwargs", '{"enable_thinking": false}']
    if vllm_args:
        for key, flag in VLLM_VALUE_FLAGS:
            value = vllm_args.get(key)
            if value is None:
                continue
            cmd.extend([flag, str(value)])
        for key, flag in VLLM_BOOL_FLAGS:
            value = vllm_args.get(key)
            if value is None:
                continue
            cmd.append(flag if bool(value) else f"--no-{flag[2:]}")
    if reasoning_parser:
        cmd.extend(["--reasoning-parser", reasoning_parser])
    if runner:
        cmd.extend(["--runner", runner])
    if convert:
        cmd.extend(["--convert", convert])

    env = os.environ.copy()
    if visible_gpus:
        env["CUDA_VISIBLE_DEVICES"] = visible_gpus

    print(
        f"{log_prefix} starting {label}: {' '.join(cmd)} "
        f"(CUDA_VISIBLE_DEVICES={visible_gpus})",
        flush=True,
    )
    path = service_log_path(kind="vllm", label=label, port=port, root=log_root)
    with path.open("ab") as log_handle:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    _SPAWNED_VLLM_SERVERS.append(proc)
    if proc.pid is not None:
        _PROCESS_LOG_PATHS[proc.pid] = path
    return (base_url, label, proc)


def cleanup_managed_vllm_servers() -> None:
    if _PRESERVE_MANAGED_VLLM_SERVERS:
        return
    for proc in _SPAWNED_VLLM_SERVERS:
        if proc.poll() is None:
            print(f"[llm] stopping vLLM server pid={proc.pid}", flush=True)
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def preserve_managed_vllm_servers() -> None:
    global _PRESERVE_MANAGED_VLLM_SERVERS
    _PRESERVE_MANAGED_VLLM_SERVERS = True


def install_signal_cleanup() -> None:
    global _SIGNAL_CLEANUP_INSTALLED
    if _SIGNAL_CLEANUP_INSTALLED:
        return

    def _handler(signum: int, _frame: Any) -> None:
        cleanup_managed_vllm_servers()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handler)
    _SIGNAL_CLEANUP_INSTALLED = True


def ensure_openai_like_endpoint(
    *,
    backend: str,
    base_url: str,
    model_name: str,
    api_key: str,
    label: str,
    visible_gpus: str = "",
    vllm_args: dict[str, Any] | None = None,
    startup_timeout_seconds: int = 600,
    log_root: Path | None = None,
    log_prefix: str = "[llm]",
) -> str:
    normalized_base_url = resolve_vllm_base_url(base_url)
    if backend == "api":
        check_endpoint_reachable_with_api_key(normalized_base_url, label, api_key=api_key)
        verify_model_served_with_api_key(normalized_base_url, model_name, label, api_key=api_key)
        return normalized_base_url

    if backend != "local":
        raise ValueError(f"Unsupported llm backend: {backend}")

    if is_endpoint_up(normalized_base_url):
        check_endpoint_reachable_with_api_key(normalized_base_url, label, api_key=api_key)
        verify_model_served_with_api_key(normalized_base_url, model_name, label, api_key=api_key)
        return normalized_base_url

    pending = spawn_vllm_server(
        model=model_name,
        port=port_from_url(normalized_base_url),
        base_url=normalized_base_url,
        visible_gpus=visible_gpus,
        label=label,
        vllm_args=vllm_args,
        log_root=log_root,
        log_prefix=log_prefix,
    )
    if pending is not None:
        wait_for_endpoints([pending], timeout_sec=startup_timeout_seconds, log_prefix=log_prefix)
    check_endpoint_reachable_with_api_key(normalized_base_url, label, api_key=api_key)
    verify_model_served_with_api_key(normalized_base_url, model_name, label, api_key=api_key)
    return normalized_base_url


def ensure_openai_like_endpoints(
    *,
    backend: str,
    base_url: str,
    model_name: str,
    api_key: str,
    label: str,
    worker_visible_gpus: list[str],
    vllm_args: dict[str, Any] | None = None,
    startup_timeout_seconds: int = 600,
    log_root: Path | None = None,
) -> list[str]:
    normalized_base_url = resolve_vllm_base_url(base_url)
    if backend == "api":
        check_endpoint_reachable_with_api_key(normalized_base_url, label, api_key=api_key)
        verify_model_served_with_api_key(normalized_base_url, model_name, label, api_key=api_key)
        return [normalized_base_url]

    if backend != "local":
        raise ValueError(f"Unsupported llm backend: {backend}")
    if not worker_visible_gpus:
        raise ValueError("worker_visible_gpus is required for local endpoint pools")

    validate_disjoint_visible_gpu_workers(worker_visible_gpus)
    validate_tensor_parallel_size_for_workers(
        tensor_parallel_size=int((vllm_args or {}).get("tensor_parallel_size", 1) or 1),
        worker_visible_gpus=worker_visible_gpus,
    )

    worker_urls = allocate_worker_base_urls(normalized_base_url, len(worker_visible_gpus))
    pending: list[tuple[str, str, subprocess.Popen[bytes]]] = []
    for index, (worker_url, visible_gpus) in enumerate(zip(worker_urls, worker_visible_gpus, strict=True), start=1):
        worker_label = label if len(worker_urls) == 1 else f"{label}[{index}]"
        if is_endpoint_up(worker_url):
            check_endpoint_reachable_with_api_key(worker_url, worker_label, api_key=api_key)
            verify_model_served_with_api_key(worker_url, model_name, worker_label, api_key=api_key)
            continue
        result = spawn_vllm_server(
            model=model_name,
            port=port_from_url(worker_url),
            base_url=worker_url,
            visible_gpus=visible_gpus,
            label=worker_label,
            vllm_args=vllm_args,
            log_root=log_root,
        )
        if result is not None:
            pending.append(result)
    if pending:
        wait_for_endpoints(pending, timeout_sec=startup_timeout_seconds)
    for index, worker_url in enumerate(worker_urls, start=1):
        worker_label = label if len(worker_urls) == 1 else f"{label}[{index}]"
        check_endpoint_reachable_with_api_key(worker_url, worker_label, api_key=api_key)
        verify_model_served_with_api_key(worker_url, model_name, worker_label, api_key=api_key)
    return worker_urls


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


install_signal_cleanup()
atexit.register(cleanup_managed_vllm_servers)
