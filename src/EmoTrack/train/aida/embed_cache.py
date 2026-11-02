"""Build and manage embedding caches for the AttnAgg method.

Extracts per-utterance embeddings from a frozen Qwen3-Embedding-8B backbone
and caches them as .pt files so the aggregator trains on pre-computed tensors.

Cache layout:
    {cache_dir}/{dataset}/{run_id}/session_{session_id}.pt

Each .pt file is a dict:
    {
        "client_embeddings": Tensor(N_client, 4096),
        "client_texts": list[str],
        "counselor_embeddings": Tensor(N_counselor, 4096),
        "counselor_texts": list[str],
    }
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


def _last_token_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Qwen3-Embedding native last-token pooling."""
    B = hidden_states.size(0)
    # Left-padded: all rows end on real tokens -> just take last
    if attention_mask[:, -1].sum() == B:
        return hidden_states[:, -1]
    # Right-padded: find last non-pad position
    seq_lengths = attention_mask.sum(dim=1)
    return hidden_states[torch.arange(B, device=hidden_states.device), seq_lengths - 1]


def _parse_transcript(
    transcript_path: Path,
) -> dict[int, dict[str, list[str]]]:
    """Parse a transcript JSONL into {session_id: {"client": [...], "counselor": [...]}}."""
    sessions: dict[int, dict[str, list[str]]] = defaultdict(
        lambda: {"client": [], "counselor": []},
    )
    for line in transcript_path.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        session_id = row.get("session_id", 1)
        speaker = row.get("speaker", "")
        text = (row.get("text", "") or "").strip()
        if not text:
            continue
        if speaker == "Counselor":
            sessions[session_id]["counselor"].append(text)
        else:
            sessions[session_id]["client"].append(text)
    return dict(sessions)


def _transcript_path_for_run(run_dir: Path) -> Path | None:
    canonical = run_dir / "common" / "transcript.jsonl"
    if canonical.is_file():
        return canonical
    legacy = run_dir / "transcript.jsonl"
    if legacy.is_file():
        return legacy
    return None


def _iter_run_transcript_paths(source_root: Path) -> list[tuple[str, Path]]:
    """Return ``(run_id, transcript_path)`` pairs for canonical result bundles."""
    runs: list[tuple[str, Path]] = []
    for child in sorted(path for path in source_root.iterdir() if path.is_dir()):
        transcript_path = _transcript_path_for_run(child)
        if transcript_path is not None:
            runs.append((child.name, transcript_path))
            continue
        for run_dir in sorted(path for path in child.iterdir() if path.is_dir()):
            transcript_path = _transcript_path_for_run(run_dir)
            if transcript_path is not None:
                runs.append((run_dir.name, transcript_path))
    return runs


@torch.no_grad()
def _embed_texts(
    texts: list[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """Embed a list of texts with the frozen backbone. Returns (N, embed_dim)."""
    if not texts:
        return torch.zeros(0, model.config.hidden_size)

    all_embeddings: list[torch.Tensor] = []
    device = next(model.parameters()).device

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        hidden = model(**encoded).last_hidden_state
        pooled = _last_token_pool(hidden, encoded["attention_mask"])
        all_embeddings.append(pooled.cpu().float())

    return torch.cat(all_embeddings, dim=0)


def build_embedding_cache(
    source_root: Path,
    cache_dir: Path,
    dataset_name: str,
    model_name: str = "Qwen/Qwen3-Embedding-8B",
    batch_size: int = 64,
    max_length: int = 512,
) -> Path:
    """Build embedding cache for a dataset.

    Args:
        source_root: Root of transcript files (e.g. result/daic_woz
            or result/longcounsel-8).
        cache_dir: Output cache directory.
        dataset_name: "longcounsel8" or "daic".
        model_name: Hugging Face model name for the frozen backbone.
        batch_size: Batch size for embedding extraction.
        max_length: Max token length per utterance.

    Returns:
        Path to the dataset cache subdirectory.
    """
    out_dir = cache_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all transcript files from canonical result bundles.
    run_transcripts = _iter_run_transcript_paths(source_root)
    if not run_transcripts:
        raise FileNotFoundError(f"No transcript.jsonl files found in {source_root}")

    logger.info(
        "Building %s embedding cache: %d runs, output=%s",
        dataset_name, len(run_transcripts), out_dir,
    )

    # Load model
    logger.info("Loading frozen backbone: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Process each run
    for run_id, transcript_path in tqdm(run_transcripts, desc=f"Caching {dataset_name}"):
        sessions = _parse_transcript(transcript_path)

        for session_id, utterances in sessions.items():
            cache_path = out_dir / run_id / f"session_{session_id}.pt"
            if cache_path.exists():
                continue

            client_texts = utterances["client"]
            counselor_texts = utterances["counselor"]

            client_emb = _embed_texts(
                client_texts, model, tokenizer, batch_size, max_length,
            )
            counselor_emb = _embed_texts(
                counselor_texts, model, tokenizer, batch_size, max_length,
            )

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "client_embeddings": client_emb,
                    "client_texts": client_texts,
                    "counselor_embeddings": counselor_emb,
                    "counselor_texts": counselor_texts,
                },
                cache_path,
            )

    logger.info("Embedding cache complete: %s", out_dir)
    return out_dir


def _embed_texts_vllm(
    texts: list[str],
    client: "openai.OpenAI",
    model_name: str,
    batch_size: int,
) -> torch.Tensor:
    """Embed texts via vLLM's OpenAI-compatible embeddings API."""
    if not texts:
        return torch.zeros(0, 4096)

    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = client.embeddings.create(input=batch, model=model_name)
        for item in resp.data:
            all_embeddings.append(item.embedding)

    return torch.tensor(all_embeddings, dtype=torch.float32)


def build_embedding_cache_vllm(
    source_root: Path,
    cache_dir: Path,
    dataset_name: str,
    base_url: str = "http://localhost:8200/v1",
    model_name: str = "Qwen/Qwen3-Embedding-8B",
    batch_size: int = 256,
) -> Path:
    """Build embedding cache using a vLLM embedding server.

    Start the server first:
        vllm serve Qwen/Qwen3-Embedding-8B --task embed \\
            --port 8200 --dtype bfloat16
    """
    import openai

    out_dir = cache_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    run_transcripts = _iter_run_transcript_paths(source_root)
    if not run_transcripts:
        raise FileNotFoundError(f"No transcript.jsonl files found in {source_root}")

    logger.info(
        "Building %s embedding cache (vLLM): %d runs, output=%s",
        dataset_name, len(run_transcripts), out_dir,
    )

    client = openai.OpenAI(base_url=base_url, api_key="EMPTY")

    for run_id, transcript_path in tqdm(run_transcripts, desc=f"Caching {dataset_name} (vLLM)"):
        sessions = _parse_transcript(transcript_path)

        for session_id, utterances in sessions.items():
            cache_path = out_dir / run_id / f"session_{session_id}.pt"
            if cache_path.exists():
                continue

            client_texts = utterances["client"]
            counselor_texts = utterances["counselor"]

            client_emb = _embed_texts_vllm(client_texts, client, model_name, batch_size)
            counselor_emb = _embed_texts_vllm(counselor_texts, client, model_name, batch_size)

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "client_embeddings": client_emb,
                    "client_texts": client_texts,
                    "counselor_embeddings": counselor_emb,
                    "counselor_texts": counselor_texts,
                },
                cache_path,
            )

    logger.info("Embedding cache complete (vLLM): %s", out_dir)
    return out_dir
