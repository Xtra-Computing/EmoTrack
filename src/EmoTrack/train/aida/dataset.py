"""Datasets for AIDA-feature models: load cached AIDA per-question scores."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from EmoTrack.generative.schema import PHQ8_SYMPTOMS as _PHQ8_SYMPTOMS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AIDA feature cache loading
# ---------------------------------------------------------------------------

def load_aida_feature_cache(path: Path) -> dict[str, list[float]]:
    """Load AIDA feature cache. Returns {row_key: features_list}."""
    cache: dict[str, list[float]] = {}

    def _ingest(entry: dict) -> None:
        row_key = entry.get("row_key", "")
        features = entry.get("features")
        if row_key and isinstance(features, list) and len(features) > 0:
            cache[row_key] = [float(f) for f in features]

    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in (payload.get("entries", []) if isinstance(payload, dict) else []):
            if isinstance(entry, dict):
                _ingest(entry)

    entry_dir = path.with_suffix("")
    if entry_dir.is_dir():
        for f in entry_dir.glob("*.json"):
            try:
                entry = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(entry, dict):
                    _ingest(entry)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping unreadable cache entry %s: %s", f, exc)
                continue

    return cache


def compute_normalization_stats(
    train_examples: list[dict],
    cache: dict[str, list[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean/std from train split only."""
    rows = []
    for ex in train_examples:
        key = f"{ex['run_id']}::s{ex['session_index']}"
        if key in cache:
            rows.append(cache[key])
    if not rows:
        n = len(next(iter(cache.values()))) if cache else 23
        return torch.zeros(n), torch.ones(n)
    X = torch.tensor(rows, dtype=torch.float32)
    return X.mean(dim=0), X.std(dim=0)


# ---------------------------------------------------------------------------
# Shared split logic
# ---------------------------------------------------------------------------

def _split_examples(
    examples: list[dict],
    split_file: str,
    split_by: str,
    eval_split: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    if split_file:
        split_data = json.loads(Path(split_file).read_text(encoding="utf-8"))
        run_splits: dict[str, str] = {}
        if isinstance(split_data.get("runs"), list):
            for entry in split_data["runs"]:
                run_splits[entry["run_id"]] = entry["split"]
        elif isinstance(split_data.get("runs"), dict):
            run_splits = split_data["runs"]
        train, val, test = [], [], []
        for ex in examples:
            s = run_splits.get(ex["run_id"], "")
            if s == "train":
                train.append(ex)
            elif s in ("val", "dev"):
                val.append(ex)
            elif s == "test":
                test.append(ex)
        return train, val, test

    rng = random.Random(seed)
    if split_by == "trend":
        groups = sorted(set(ex.get("trend_name", ex["run_id"]) for ex in examples))
    else:
        groups = sorted(set(ex["run_id"] for ex in examples))
    rng.shuffle(groups)
    n_eval = max(1, int(len(groups) * eval_split))
    eval_groups = set(groups[:n_eval])
    train, val = [], []
    for ex in examples:
        key = ex.get("trend_name", ex["run_id"]) if split_by == "trend" else ex["run_id"]
        (val if key in eval_groups else train).append(ex)
    return train, val, []


def _filter_to_cache(examples: list[dict], cache: dict[str, list[float]]) -> list[dict]:
    filtered = [ex for ex in examples if f"{ex['run_id']}::s{ex['session_index']}" in cache]
    if len(filtered) < len(examples):
        logger.warning(
            "Filtered %d/%d examples missing from AIDA cache",
            len(examples) - len(filtered), len(examples),
        )
    return filtered


def _load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _deterministic_loader_kwargs(config, *, shuffle: bool) -> dict:
    if not getattr(config, "deterministic_training", False):
        return {}
    from EmoTrack.utils.determinism import make_loader_generator, seed_worker

    kwargs = {"worker_init_fn": seed_worker}
    if shuffle:
        kwargs["generator"] = make_loader_generator(int(config.seed))
    return kwargs


# ---------------------------------------------------------------------------
# AttnAgg dataset
# ---------------------------------------------------------------------------

class AidaAttnAggDataset(Dataset):
    def __init__(self, examples: list[dict], cache: dict[str, list[float]]) -> None:
        self._examples = examples
        self._cache = cache

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self._examples[idx]
        key = f"{ex['run_id']}::s{ex['session_index']}"
        features = torch.tensor(self._cache[key], dtype=torch.float32)
        target = torch.tensor(
            [float(ex["target_phq_items"][s]) for s in _PHQ8_SYMPTOMS],
            dtype=torch.float32,
        )
        return {"features": features, "target": target, "session_index": ex["session_index"]}


def _collate_attnagg(batch: list[dict]) -> dict:
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "session_index": torch.tensor([b["session_index"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# GRU dataset
# ---------------------------------------------------------------------------

class AidaGRUDataset(Dataset):
    def __init__(self, examples: list[dict], cache: dict[str, list[float]]) -> None:
        self._examples = examples
        self._cache = cache

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self._examples[idx]
        run_id = ex["run_id"]
        session_index = ex["session_index"]
        key = f"{run_id}::s{session_index}"
        cur = torch.tensor(self._cache[key], dtype=torch.float32)
        target = torch.tensor(
            [float(ex["target_phq_items"][s]) for s in _PHQ8_SYMPTOMS],
            dtype=torch.float32,
        )

        hist_feats, hist_lags = [], []
        is_daic = run_id.isdigit()
        if not is_daic:
            for prior in range(1, session_index):
                pk = f"{run_id}::s{prior}"
                if pk in self._cache:
                    hist_feats.append(torch.tensor(self._cache[pk], dtype=torch.float32))
                    hist_lags.append(session_index - prior)

        return {
            "current_features": cur,
            "history_features": hist_feats,
            "history_lags": hist_lags,
            "target": target,
            "session_index": session_index,
        }


def _collate_gru(batch: list[dict]) -> dict:
    B = len(batch)
    cur = torch.stack([b["current_features"] for b in batch])
    targets = torch.stack([b["target"] for b in batch])
    sess = torch.tensor([b["session_index"] for b in batch])

    all_hist, all_owners, all_lags = [], [], []
    hist_len = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        H = len(b["history_features"])
        hist_len[i] = H
        for j in range(H):
            all_hist.append(b["history_features"][j])
            all_owners.append(i)
            all_lags.append(b["history_lags"][j])

    if all_hist:
        hist_features = torch.stack(all_hist)
        hist_owner = torch.tensor(all_owners, dtype=torch.long)
        hist_lag = torch.tensor(all_lags, dtype=torch.long)
    else:
        hist_features = hist_owner = hist_lag = None

    return {
        "cur_features": cur,
        "hist_features": hist_features,
        "hist_owner": hist_owner,
        "hist_lag": hist_lag,
        "hist_len": hist_len,
        "target": targets,
        "session_index": sess,
    }


# ---------------------------------------------------------------------------
# StackAttn dataset
# ---------------------------------------------------------------------------

class AidaStackAttnDataset(Dataset):
    def __init__(self, examples: list[dict], cache: dict[str, list[float]]) -> None:
        self._examples = examples
        self._cache = cache

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self._examples[idx]
        run_id = ex["run_id"]
        session_index = ex["session_index"]
        key = f"{run_id}::s{session_index}"
        cur = torch.tensor(self._cache[key], dtype=torch.float32)
        target = torch.tensor(
            [float(ex["target_phq_items"][s]) for s in _PHQ8_SYMPTOMS],
            dtype=torch.float32,
        )

        prev = None
        is_daic = run_id.isdigit()
        if not is_daic and session_index > 1:
            pk = f"{run_id}::s{session_index - 1}"
            if pk in self._cache:
                prev = torch.tensor(self._cache[pk], dtype=torch.float32)

        return {
            "current_features": cur,
            "previous_features": prev,
            "target": target,
            "session_index": session_index,
        }


def _collate_stackattn(batch: list[dict]) -> dict:
    B = len(batch)
    F = batch[0]["current_features"].size(0)
    cur = torch.stack([b["current_features"] for b in batch])
    targets = torch.stack([b["target"] for b in batch])
    sess = torch.tensor([b["session_index"] for b in batch])

    prev = torch.zeros(B, F)
    has_prev = torch.zeros(B, dtype=torch.bool)
    for i, b in enumerate(batch):
        if b["previous_features"] is not None:
            prev[i] = b["previous_features"]
            has_prev[i] = True

    return {
        "cur_features": cur,
        "prev_features": prev,
        "has_prev": has_prev,
        "target": targets,
        "session_index": sess,
    }


# ---------------------------------------------------------------------------
# Dataloader builders
# ---------------------------------------------------------------------------

def _merge_extra_cache(
    cache: dict[str, list[float]],
    extra_path: str,
) -> dict[str, list[float]]:
    """Concatenate extra features onto existing cache entries in-place."""
    extra = load_aida_feature_cache(Path(extra_path))
    merged: dict[str, list[float]] = {}
    for key in cache:
        if key in extra:
            merged[key] = cache[key] + extra[key]
    logger.info(
        "Merged %d/%d entries with extra features (%d+%d=%d dims)",
        len(merged), len(cache),
        len(next(iter(cache.values()))) if cache else 0,
        len(next(iter(extra.values()))) if extra else 0,
        len(next(iter(merged.values()))) if merged else 0,
    )
    return merged


def _prepare_common(config, collate_fn, dataset_cls):
    all_examples = _load_jsonl(Path(config.dataset_path))
    cache = load_aida_feature_cache(Path(config.aida_cache_path))
    if config.aida_extra_cache_path:
        cache = _merge_extra_cache(cache, config.aida_extra_cache_path)
    logger.info("Loaded %d examples, %d cached AIDA features", len(all_examples), len(cache))

    train_ex, val_ex, test_ex = _split_examples(
        all_examples, config.split_file, config.split_by, config.eval_split, config.seed,
    )
    train_ex = _filter_to_cache(train_ex, cache)
    val_ex = _filter_to_cache(val_ex, cache)
    test_ex = _filter_to_cache(test_ex, cache)

    norm_mean, norm_std = compute_normalization_stats(train_ex, cache)

    if config.daic_oversample > 1:
        daic = [ex for ex in train_ex if ex["run_id"].isdigit()]
        if daic:
            train_ex = train_ex + daic * (config.daic_oversample - 1)

    logger.info("Train: %d, Val: %d, Test: %d", len(train_ex), len(val_ex), len(test_ex))

    train_loader = DataLoader(
        dataset_cls(train_ex, cache), batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=2,
        **_deterministic_loader_kwargs(config, shuffle=True),
    )
    val_loader = DataLoader(
        dataset_cls(val_ex, cache), batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=2,
        **_deterministic_loader_kwargs(config, shuffle=False),
    )
    test_loader = None
    if test_ex:
        test_loader = DataLoader(
            dataset_cls(test_ex, cache), batch_size=config.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=2,
            **_deterministic_loader_kwargs(config, shuffle=False),
        )
    return train_loader, val_loader, test_loader, norm_mean, norm_std


def prepare_dataloaders_attnagg(config):
    return _prepare_common(config, _collate_attnagg, AidaAttnAggDataset)


# ---------------------------------------------------------------------------
# Hybrid AIDA + sentence-embedding dataset
# ---------------------------------------------------------------------------

_SPEAKER_VIEWS = ("client", "counselor", "ordered")
_INPUT_MODES = ("full", "aida_only", "embedding_only")


def _select_session_embeddings(
    cached: dict,
    speaker_view: str,
    transcript_path: Path | None,
    session_index: int,
) -> torch.Tensor:
    if speaker_view == "client":
        return cached["client_embeddings"]
    if speaker_view == "counselor":
        return cached["counselor_embeddings"]
    if speaker_view != "ordered":
        raise ValueError(f"Unknown speaker_view: {speaker_view!r}")

    if transcript_path is None or not transcript_path.is_file():
        raise FileNotFoundError(
            f"speaker_view='ordered' requires transcript at {transcript_path}",
        )

    client_emb = cached["client_embeddings"]
    counselor_emb = cached["counselor_embeddings"]
    embed_dim = client_emb.size(-1) if client_emb.numel() else counselor_emb.size(-1)

    client_idx = 0
    counselor_idx = 0
    ordered: list[torch.Tensor] = []
    with transcript_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row.get("session_id", 1)) != session_index:
                continue
            if not (row.get("text", "") or "").strip():
                continue
            if row.get("speaker", "") == "Counselor":
                if counselor_idx < counselor_emb.size(0):
                    ordered.append(counselor_emb[counselor_idx])
                counselor_idx += 1
            else:
                if client_idx < client_emb.size(0):
                    ordered.append(client_emb[client_idx])
                client_idx += 1

    if not ordered:
        return torch.zeros(0, embed_dim)
    return torch.stack(ordered, dim=0)


class AidaEmbedAttnAggDataset(Dataset):
    def __init__(
        self,
        examples: list[dict],
        cache: dict[str, list[float]],
        embed_cache_dir: Path,
        max_sentences: int = 384,
        backbone_embed_dim: int = 4096,
        input_mode: str = "full",
        use_prev_sent: bool = False,
        speaker_view: str = "client",
        psyche_transcript_root: Path | None = None,
        daic_transcript_root: Path | None = None,
    ) -> None:
        if speaker_view not in _SPEAKER_VIEWS:
            raise ValueError(
                f"speaker_view must be one of {_SPEAKER_VIEWS}, got {speaker_view!r}",
            )
        if input_mode not in _INPUT_MODES:
            raise ValueError(f"input_mode must be one of {_INPUT_MODES}, got {input_mode!r}")
        self._examples = examples
        self._cache = cache
        self._embed_dir = embed_cache_dir
        self._max_sentences = max_sentences
        self._backbone_embed_dim = backbone_embed_dim
        self._input_mode = input_mode
        self._use_prev_sent = use_prev_sent
        self._speaker_view = speaker_view
        self._psyche_transcript_root = psyche_transcript_root
        self._daic_transcript_root = daic_transcript_root

    def __len__(self) -> int:
        return len(self._examples)

    def _transcript_path(self, run_id: str) -> Path | None:
        if self._speaker_view != "ordered":
            return None
        is_daic = run_id.isdigit()
        root = self._daic_transcript_root if is_daic else self._psyche_transcript_root
        if root is None:
            dataset = "DAIC" if is_daic else "LongCounsel-8"
            raise ValueError(
                f"speaker_view='ordered' requires a transcript root for {dataset}",
            )
        return root / run_id / "common" / "transcript.jsonl"

    def __getitem__(self, idx: int) -> dict:
        ex = self._examples[idx]
        key = f"{ex['run_id']}::s{ex['session_index']}"
        features = torch.tensor(self._cache[key], dtype=torch.float32)
        target = torch.tensor(
            [float(ex["target_phq_items"][s]) for s in _PHQ8_SYMPTOMS],
            dtype=torch.float32,
        )

        is_daic = ex["run_id"].isdigit()
        subdir = "daic" if is_daic else "longcounsel8"
        transcript_path = None
        if self._input_mode == "aida_only":
            emb = torch.zeros(0, self._backbone_embed_dim)
            n = 0
        else:
            transcript_path = self._transcript_path(ex["run_id"])
            emb_path = self._embed_dir / subdir / ex["run_id"] / f"session_{ex['session_index']}.pt"
            cached = torch.load(emb_path, map_location="cpu", weights_only=False)
            emb = _select_session_embeddings(
                cached,
                self._speaker_view,
                transcript_path,
                ex["session_index"],
            )
            n = min(emb.size(0), self._max_sentences)
            emb = emb[:n]

        result = {
            "features": features,
            "sent_embeddings": emb,
            "sent_n": n,
            "target": target,
            "session_index": ex["session_index"],
        }

        if self._use_prev_sent:
            prev_emb = None
            prev_n = 0
            if not is_daic and ex["session_index"] > 1:
                prev_path = self._embed_dir / subdir / ex["run_id"] / f"session_{ex['session_index'] - 1}.pt"
                if prev_path.exists():
                    prev_cached = torch.load(prev_path, map_location="cpu", weights_only=False)
                    prev_emb_full = _select_session_embeddings(
                        prev_cached,
                        self._speaker_view,
                        transcript_path,
                        ex["session_index"] - 1,
                    )
                    prev_n = min(prev_emb_full.size(0), self._max_sentences)
                    prev_emb = prev_emb_full[:prev_n]
            result["prev_sent_embeddings"] = prev_emb
            result["prev_sent_n"] = prev_n

        return result


def _collate_embed_attnagg(batch: list[dict]) -> dict:
    features = torch.stack([b["features"] for b in batch])
    targets = torch.stack([b["target"] for b in batch])
    session_indices = torch.tensor([b["session_index"] for b in batch])

    max_n = max(b["sent_n"] for b in batch)
    embed_dim = batch[0]["sent_embeddings"].size(-1)
    B = len(batch)
    sent = torch.zeros(B, max_n, embed_dim)
    mask = torch.zeros(B, max_n)
    for i, b in enumerate(batch):
        n = b["sent_n"]
        sent[i, :n] = b["sent_embeddings"]
        mask[i, :n] = 1.0

    result = {
        "features": features,
        "sent_embeddings": sent,
        "sent_mask": mask,
        "target": targets,
        "session_index": session_indices,
    }

    # Previous-session sentence embeddings
    if "prev_sent_embeddings" in batch[0]:
        has_any = any(b["prev_sent_n"] > 0 for b in batch)
        if has_any:
            max_prev_n = max(b["prev_sent_n"] for b in batch)
            max_prev_n = max(max_prev_n, 1)
            prev_sent = torch.zeros(B, max_prev_n, embed_dim)
            prev_sent_mask = torch.zeros(B, max_prev_n)
            for i, b in enumerate(batch):
                pn = b["prev_sent_n"]
                if pn > 0 and b["prev_sent_embeddings"] is not None:
                    prev_sent[i, :pn] = b["prev_sent_embeddings"]
                    prev_sent_mask[i, :pn] = 1.0
        else:
            prev_sent = torch.zeros(B, 1, embed_dim)
            prev_sent_mask = torch.zeros(B, 1)
        result["prev_sent_embeddings"] = prev_sent
        result["prev_sent_mask"] = prev_sent_mask

    return result


def _require_cached_examples(
    examples, embed_cache_dir, cache, *, require_embed_cache: bool
):
    """Require every example's features instead of changing the experimental split."""
    missing = []
    for example in examples:
        key = f"{example['run_id']}::s{example['session_index']}"
        if key not in cache:
            missing.append(f"AIDA features: {key}")
        if require_embed_cache:
            dataset = "daic" if example["run_id"].isdigit() else "longcounsel8"
            path = (
                embed_cache_dir
                / dataset
                / example["run_id"]
                / f"session_{example['session_index']}.pt"
            )
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} feature/embedding cache entries; first: {missing[:5]}. "
            "Run the prepare and features stages before training."
        )


def prepare_dataloaders_embed_attnagg(config):
    all_examples = _load_jsonl(Path(config.dataset_path))
    cache = load_aida_feature_cache(Path(config.aida_cache_path))
    if config.aida_extra_cache_path:
        cache = _merge_extra_cache(cache, config.aida_extra_cache_path)

    embed_dir = Path(config.embed_cache_dir)
    logger.info("Loaded %d examples, %d cached AIDA features", len(all_examples), len(cache))

    train_ex, val_ex, test_ex = _split_examples(
        all_examples, config.split_file, config.split_by, config.eval_split, config.seed,
    )
    input_mode = getattr(config, "input_mode", "full")
    require_embed_cache = input_mode != "aida_only"
    _require_cached_examples(train_ex, embed_dir, cache, require_embed_cache=require_embed_cache)
    _require_cached_examples(val_ex, embed_dir, cache, require_embed_cache=require_embed_cache)
    _require_cached_examples(test_ex, embed_dir, cache, require_embed_cache=require_embed_cache)

    norm_mean, norm_std = compute_normalization_stats(train_ex, cache)

    if config.daic_oversample > 1:
        daic = [ex for ex in train_ex if ex["run_id"].isdigit()]
        if daic:
            train_ex = train_ex + daic * (config.daic_oversample - 1)

    logger.info("Train: %d, Val: %d, Test: %d", len(train_ex), len(val_ex), len(test_ex))

    use_prev_sent = getattr(config, "prev_sent_mode", "none") != "none"
    speaker_view = getattr(config, "speaker_view", "client")
    psyche_root = Path(config.psyche_transcript_root) if getattr(config, "psyche_transcript_root", "") else None
    daic_root = Path(config.daic_transcript_root) if getattr(config, "daic_transcript_root", "") else None
    if speaker_view == "ordered" and psyche_root is None and daic_root is None:
        raise ValueError(
            "speaker_view='ordered' requires psyche_transcript_root and/or "
            "daic_transcript_root to locate transcript.jsonl files."
        )
    mk = lambda exs: AidaEmbedAttnAggDataset(
        exs,
        cache,
        embed_dir,
        config.max_sentences,
        backbone_embed_dim=config.backbone_embed_dim,
        input_mode=input_mode,
        use_prev_sent=use_prev_sent,
        speaker_view=speaker_view,
        psyche_transcript_root=psyche_root,
        daic_transcript_root=daic_root,
    )
    train_loader = DataLoader(
        mk(train_ex), batch_size=config.batch_size,
        shuffle=True, collate_fn=_collate_embed_attnagg, num_workers=2,
        **_deterministic_loader_kwargs(config, shuffle=True),
    )
    val_loader = DataLoader(
        mk(val_ex), batch_size=config.batch_size,
        shuffle=False, collate_fn=_collate_embed_attnagg, num_workers=2,
        **_deterministic_loader_kwargs(config, shuffle=False),
    )
    test_loader = None
    if test_ex:
        test_loader = DataLoader(
            mk(test_ex), batch_size=config.batch_size,
            shuffle=False, collate_fn=_collate_embed_attnagg, num_workers=2,
            **_deterministic_loader_kwargs(config, shuffle=False),
        )
    return train_loader, val_loader, test_loader, norm_mean, norm_std


def prepare_dataloaders_gru(config):
    return _prepare_common(config, _collate_gru, AidaGRUDataset)


def prepare_dataloaders_stackattn(config):
    return _prepare_common(config, _collate_stackattn, AidaStackAttnDataset)
