"""Training loops for AIDA-feature models."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from EmoTrack.generative.schema import PHQ8_SYMPTOMS as _PHQ8_SYMPTOMS
from EmoTrack.generative.schema import phq8_severity_band
from EmoTrack.train.aida.config import (
    AidaAttnAggConfig,
    AidaEmbedAttnAggConfig,
    AidaEmbedHybridConfig,
    AidaGRUConfig,
    AidaStackAttnConfig,
)
from EmoTrack.train.aida.dataset import (
    prepare_dataloaders_attnagg,
    prepare_dataloaders_embed_attnagg,
    prepare_dataloaders_gru,
    prepare_dataloaders_stackattn,
)
from EmoTrack.train.aida.model import (
    _HYBRID_MODEL,
    AidaAttnAgg,
    AidaEmbedAttnAgg,
    AidaEmbedHybrid,
    AidaGRU,
    AidaStackAttn,
    load_checkpoint_attnagg,
    save_checkpoint_attnagg,
    save_checkpoint_embed_attnagg,
    save_checkpoint_gru,
    save_checkpoint_hybrid,
    save_checkpoint_stackattn,
)
from EmoTrack.utils.paths import resolve_seed_cache_path

logger = logging.getLogger(__name__)


def _lr_schedule(warmup: int, total: int):
    def fn(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))
    return fn


def _maybe_enable_deterministic_training(config) -> None:
    if not getattr(config, "deterministic_training", False):
        return
    from EmoTrack.utils.determinism import set_determinism

    set_determinism(int(config.seed))
    logger.info("Deterministic training enabled with seed=%d", int(config.seed))


def _ordinal_nll(class_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    t = targets.long().clamp(0, 3)
    log_p = class_probs.clamp(min=1e-7).log()
    return -log_p.gather(-1, t.unsqueeze(-1)).squeeze(-1).mean()


def _false_negative_loss(
    pred_total: torch.Tensor, target_total: torch.Tensor, threshold: float,
) -> torch.Tensor:
    positive = target_total >= threshold
    penalty = torch.relu(threshold - pred_total)
    return torch.where(positive, penalty, torch.zeros_like(penalty)).mean()


def _total_item_loss(
    expected: torch.Tensor,
    target: torch.Tensor,
    huber_fn: nn.HuberLoss,
    config,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_total = expected.sum(dim=1)
    target_total = target.sum(dim=1)
    total_huber = huber_fn(pred_total, target_total)
    item_huber = huber_fn(expected, target)
    loss = config.total_huber_weight * (
        total_huber + config.item_loss_weight * item_huber
    )
    loss += getattr(config, "fn_loss_weight", 0.0) * _false_negative_loss(
        pred_total, target_total, getattr(config, "fn_threshold", 10.0),
    )
    return loss, item_huber


def _evaluate(model, loader, device, config, *, forward_fn) -> dict:
    model.eval()
    total_ordinal = 0.0
    total_huber = 0.0
    total_item_huber = 0.0
    total_mae = 0.0
    total_fn_penalty = 0.0
    positives = negatives = false_negatives = false_positives = 0
    per_item_ae = [0.0] * len(_PHQ8_SYMPTOMS)
    band_correct = 0
    n = 0
    session_correct: dict[int, int] = defaultdict(int)
    session_count: dict[int, int] = defaultdict(int)
    session_ae: dict[int, float] = defaultdict(float)
    huber_fn = nn.HuberLoss(reduction="sum")

    with torch.no_grad():
        for batch in loader:
            target = batch["target"].to(device)
            expected, class_probs = forward_fn(batch, device)
            B = target.size(0)

            pred_total = expected.sum(dim=1)
            target_total = target.sum(dim=1)
            threshold = getattr(config, "fn_threshold", 10.0)
            positive = target_total >= threshold
            predicted_positive = pred_total >= threshold
            positives += positive.sum().item()
            negatives += (~positive).sum().item()
            false_negatives += (positive & ~predicted_positive).sum().item()
            false_positives += (~positive & predicted_positive).sum().item()
            total_fn_penalty += _false_negative_loss(pred_total, target_total, threshold).item() * B
            h = huber_fn(pred_total, target_total) / B
            total_huber += h.item() * B
            ih = huber_fn(expected, target) / (B * len(_PHQ8_SYMPTOMS))
            total_item_huber += ih.item() * B

            if class_probs is not None and config.ordinal_nll_weight > 0:
                total_ordinal += _ordinal_nll(class_probs, target).item() * B

            ae = (expected - target).abs().float()
            total_mae += ae.sum().item()
            for i in range(len(_PHQ8_SYMPTOMS)):
                per_item_ae[i] += ae[:, i].sum().item()

            for pt, tt, si in zip(
                pred_total.float().tolist(),
                target_total.float().tolist(),
                batch["session_index"].tolist(),
            ):
                session_count[si] += 1
                session_ae[si] += abs(pt - tt)
                if phq8_severity_band(pt) == phq8_severity_band(tt):
                    session_correct[si] += 1
                    band_correct += 1
            n += B

    J = len(_PHQ8_SYMPTOMS)
    eval_loss = (
        config.ordinal_nll_weight * total_ordinal / n
        + config.total_huber_weight * (
            total_huber / n + config.item_loss_weight * total_item_huber / n
        )
    )
    eval_loss += getattr(config, "fn_loss_weight", 0.0) * total_fn_penalty / n
    item_mae = total_mae / (n * J)
    return {
        "eval_fn_penalty": round(total_fn_penalty / n, 4),
        "eval_fnr": false_negatives / positives if positives else None,
        "eval_fpr": false_positives / negatives if negatives else None,
        "eval_loss": round(eval_loss, 4),
        "eval_mae": round(item_mae, 4),
        "eval_mae_item": round(item_mae, 4),
        "eval_mae_total": round(sum(session_ae.values()) / n, 4),
        "eval_huber_item": round(total_item_huber / n, 4),
        "eval_band_acc": round(band_correct / n, 4),
        "eval_per_item_mae": {s: round(per_item_ae[i] / n, 4) for i, s in enumerate(_PHQ8_SYMPTOMS)},
        "eval_per_session_band_acc": {
            str(si): round(session_correct[si] / session_count[si], 4)
            for si in sorted(session_count)
        },
        "eval_per_session_mae": {
            str(si): round(session_ae[si] / session_count[si], 4)
            for si in sorted(session_count)
        },
    }


def _train_loop(
    model,
    config,
    train_loader,
    val_loader,
    test_loader,
    device,
    save_fn,
    forward_fn,
    model_filename: str,
) -> Path:
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.num_epochs
    warmup = int(total_steps * config.warmup_ratio)
    scheduler = LambdaLR(optimizer, _lr_schedule(warmup, total_steps))
    huber_fn = nn.HuberLoss()

    result_dir = Path("result/psyche_eval") / config.result_name / f"seed{config.inference_seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = result_dir / "metrics.jsonl"
    metrics_file = open(metrics_path, "w", encoding="utf-8")  # noqa: SIM115
    try:
        return _train_loop_body(
            model, config, train_loader, val_loader, test_loader,
            device, save_fn, forward_fn, model_filename, metrics_file,
            optimizer, scheduler, huber_fn,
        )
    finally:
        metrics_file.close()


def _train_loop_body(
    model, config, train_loader, val_loader, test_loader,
    device, save_fn, forward_fn, model_filename, metrics_file,
    optimizer, scheduler, huber_fn,
) -> Path:
    best_dir = Path(config.output_dir) / "best"
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_train = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")
        for batch in progress:
            target = batch["target"].to(device)
            expected, class_probs = forward_fn(batch, device)

            total_item_loss, _item_huber = _total_item_loss(expected, target, huber_fn, config)

            if class_probs is not None and config.ordinal_nll_weight > 0:
                loss = config.ordinal_nll_weight * _ordinal_nll(class_probs, target) + total_item_loss
            else:
                loss = total_item_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            B = target.size(0)
            epoch_loss += loss.item() * B
            n_train += B
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / n_train
        eval_metrics = _evaluate(model, val_loader, device, config, forward_fn=forward_fn)
        eval_loss = eval_metrics["eval_loss"]

        record = {"epoch": epoch, "train_loss": round(train_loss, 4), **eval_metrics}
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()
        logger.info(
            "Epoch %d: train=%.4f eval=%.4f mae_total=%.4f band_acc=%.4f",
            epoch, train_loss, eval_loss, eval_metrics["eval_mae_total"], eval_metrics["eval_band_acc"],
        )

        if eval_loss < best_loss:
            best_loss = eval_loss
            patience_counter = 0
            save_fn(model, best_dir, config)
            logger.info("  -> New best (%.4f)", eval_loss)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if test_loader is not None:
        logger.info("Evaluating on test set...")
        best_state = torch.load(best_dir / model_filename, map_location=device, weights_only=True)
        model.load_state_dict(best_state)
        test_metrics = _evaluate(model, test_loader, device, config, forward_fn=forward_fn)
        metrics_file.write(json.dumps({"split": "test", **test_metrics}) + "\n")
        logger.info("Test: mae_total=%.4f band_acc=%.4f", test_metrics["eval_mae_total"], test_metrics["eval_band_acc"])

    logger.info("Checkpoint: %s", best_dir)
    return best_dir


# ---------------------------------------------------------------------------
# AttnAgg
# ---------------------------------------------------------------------------

def train_attnagg(config: AidaAttnAggConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _maybe_enable_deterministic_training(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AidaAttnAgg(config).to(device)
    logger.info("AidaAttnAgg params: %s", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_loader, val_loader, test_loader, norm_mean, norm_std = prepare_dataloaders_attnagg(config)
    model.tokenizer.set_normalization(norm_mean.to(device), norm_std.to(device))

    def fwd(batch, dev):
        return model(batch["features"].to(dev))

    return _train_loop(model, config, train_loader, val_loader, test_loader, device,
                       save_checkpoint_attnagg, fwd, "aida_attnagg_model.pt")


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

def train_gru(config: AidaGRUConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _maybe_enable_deterministic_training(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AidaGRU(config).to(device)

    if config.aida_attnagg_checkpoint:
        ckpt = load_checkpoint_attnagg(config.aida_attnagg_checkpoint)
        model.aggregator.load_state_dict(ckpt.state_dict(), strict=False)
        if config.freeze_aggregator:
            for p in model.aggregator.parameters():
                p.requires_grad = False
            logger.info("Loaded and froze aggregator from %s", config.aida_attnagg_checkpoint)
        else:
            logger.info("Loaded warm-start aggregator from %s", config.aida_attnagg_checkpoint)

    logger.info("AidaGRU params: %s", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_loader, val_loader, test_loader, norm_mean, norm_std = prepare_dataloaders_gru(config)
    model.aggregator.tokenizer.set_normalization(norm_mean.to(device), norm_std.to(device))

    def fwd(batch, dev):
        hf = batch["hist_features"]
        ho = batch["hist_owner"]
        hl = batch["hist_lag"]
        return model(
            batch["cur_features"].to(dev),
            hf.to(dev) if hf is not None else None,
            ho.to(dev) if ho is not None else None,
            hl.to(dev) if hl is not None else None,
            batch["hist_len"].to(dev),
        )

    return _train_loop(model, config, train_loader, val_loader, test_loader, device,
                       save_checkpoint_gru, fwd, "aida_gru_model.pt")


# ---------------------------------------------------------------------------
# StackAttn
# ---------------------------------------------------------------------------

def train_stackattn(config: AidaStackAttnConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Apply seed subfolder, resolve seed-specific AIDA cache
    config.output_dir = str(Path(config.output_dir) / f"seed{config.inference_seed}")
    if config.inference_seed != 0:
        config.seed = config.inference_seed
    config.aida_cache_path = resolve_seed_cache_path(config.aida_cache_path, config.inference_seed)
    logger.info("inference_seed=%d → seed=%d, output_dir=%s, aida_cache=%s",
                 config.inference_seed, config.seed, config.output_dir, config.aida_cache_path)

    _maybe_enable_deterministic_training(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AidaStackAttn(config).to(device)

    if config.aida_attnagg_checkpoint:
        ckpt = load_checkpoint_attnagg(config.aida_attnagg_checkpoint)
        model.aggregator.load_state_dict(ckpt.state_dict(), strict=False)
        if config.freeze_aggregator:
            for p in model.aggregator.parameters():
                p.requires_grad = False
            logger.info("Loaded and froze aggregator from %s", config.aida_attnagg_checkpoint)
        else:
            logger.info("Loaded warm-start aggregator from %s", config.aida_attnagg_checkpoint)

    logger.info("AidaStackAttn params: %s", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_loader, val_loader, test_loader, norm_mean, norm_std = prepare_dataloaders_stackattn(config)
    model.aggregator.tokenizer.set_normalization(norm_mean.to(device), norm_std.to(device))

    def fwd(batch, dev):
        return model(
            batch["cur_features"].to(dev),
            batch["prev_features"].to(dev),
            batch["has_prev"].to(dev),
        )

    return _train_loop(model, config, train_loader, val_loader, test_loader, device,
                       save_checkpoint_stackattn, fwd, "aida_stackattn_model.pt")


# ---------------------------------------------------------------------------
# Hybrid: EmbedAttnAgg
# ---------------------------------------------------------------------------

def train_embed_attnagg(config: AidaEmbedAttnAggConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _maybe_enable_deterministic_training(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AidaEmbedAttnAgg(config).to(device)
    logger.info("AidaEmbedAttnAgg params: %s", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_loader, val_loader, test_loader, norm_mean, norm_std = prepare_dataloaders_embed_attnagg(config)
    model.tokenizer.set_normalization(norm_mean.to(device), norm_std.to(device))

    def fwd(batch, dev):
        return model(
            batch["features"].to(dev),
            batch["sent_embeddings"].to(dev),
            batch["sent_mask"].to(dev),
        )

    return _train_loop(model, config, train_loader, val_loader, test_loader, device,
                       save_checkpoint_embed_attnagg, fwd, "aida_embed_attnagg_model.pt")


# ---------------------------------------------------------------------------
# Hybrid: EmbedHybrid (full sentence tokens + AIDA)
# ---------------------------------------------------------------------------

def train_hybrid(config: AidaEmbedHybridConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Apply inference_seed: nest output under seed subfolder, resolve seed-specific AIDA cache
    config.output_dir = str(Path(config.output_dir) / f"seed{config.inference_seed}")
    if config.inference_seed != 0:
        config.seed = config.inference_seed
    config.aida_cache_path = resolve_seed_cache_path(config.aida_cache_path, config.inference_seed)
    logger.info("inference_seed=%d → seed=%d, output_dir=%s, aida_cache=%s",
                 config.inference_seed, config.seed, config.output_dir, config.aida_cache_path)

    _maybe_enable_deterministic_training(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AidaEmbedHybrid(config).to(device)
    logger.info("AidaEmbedHybrid params: %s",
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Warmstart: load compatible weights from a shallower checkpoint
    if config.warmstart_checkpoint:
        ws_path = Path(config.warmstart_checkpoint)
        ws_state = torch.load(ws_path / _HYBRID_MODEL, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(ws_state, strict=False)
        logger.info("Warmstart from %s: %d missing, %d unexpected keys",
                     ws_path, len(missing), len(unexpected))
        if missing:
            logger.info("  Missing (new layers): %s", missing[:10])

    train_loader, val_loader, test_loader, norm_mean, norm_std = prepare_dataloaders_embed_attnagg(config)
    model.tokenizer.set_normalization(norm_mean.to(device), norm_std.to(device))

    def fwd(batch, dev):
        kwargs = {}
        if "prev_sent_embeddings" in batch:
            kwargs["prev_sent_embeddings"] = batch["prev_sent_embeddings"].to(dev)
            kwargs["prev_sent_mask"] = batch["prev_sent_mask"].to(dev)
        return model(
            batch["features"].to(dev),
            batch["sent_embeddings"].to(dev),
            batch["sent_mask"].to(dev),
            **kwargs,
        )

    return _train_loop(model, config, train_loader, val_loader, test_loader, device,
                       save_checkpoint_hybrid, fwd, _HYBRID_MODEL)
