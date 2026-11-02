"""AIDA-feature attention models for PHQ-8 prediction.

Architectures that use 23 AIDA per-question scalar features as input tokens:

    AidaAttnAgg    — item queries attend over 23 feature tokens
    AidaGRU        — + temporal GRU over per-item session summaries
    AidaStackAttn  — + cross-attention to immediately prior session
    AidaEmbedHybrid — full sentence tokens + AIDA tokens (full_seq fusion)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from EmoTrack.train.aida.config import (
    AidaAttnAggConfig,
    AidaEmbedAttnAggConfig,
    AidaGRUConfig,
    AidaStackAttnConfig,
)

logger = logging.getLogger(__name__)

_NUM_LEVELS = 4
_NUM_THRESHOLDS = _NUM_LEVELS - 1
_MAX_SCORE = float(_NUM_LEVELS - 1)

_ATTNAGG_CONFIG = "aida_attnagg_config.json"
_ATTNAGG_MODEL = "aida_attnagg_model.pt"
_GRU_CONFIG = "aida_gru_config.json"
_GRU_MODEL = "aida_gru_model.pt"
_STACKATTN_CONFIG = "aida_stackattn_config.json"
_STACKATTN_MODEL = "aida_stackattn_model.pt"


# ---------------------------------------------------------------------------
# Feature tokenizer
# ---------------------------------------------------------------------------

class AidaFeatureTokenizer(nn.Module):
    """Convert (B, F) raw AIDA scalars to (B, F, d_model) token embeddings.

    token_i = scalar_mlp(z_score_i)

    Each score is z-normalized then projected through a small MLP, then
    augmented with learned question/group identity embeddings.
    """

    def __init__(
        self,
        num_features: int,
        group_sizes: list[int],
        d_model: int,
        scalar_mlp_hidden: int,
        use_feature_identity: bool = True,
        skip_mode: str = "none",
        identity_mode: str = "both",
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.use_feature_identity = use_feature_identity
        self.skip_mode = skip_mode
        self.identity_mode = identity_mode

        self.scalar_mlp = nn.Sequential(
            nn.Linear(1, scalar_mlp_hidden),
            nn.GELU(),
            nn.Linear(scalar_mlp_hidden, d_model),
        )
        if skip_mode == "additive":
            self.raw_proj = nn.Linear(1, d_model)
        elif skip_mode == "gated":
            self.raw_proj = nn.Linear(1, d_model)
            gate_in = d_model + (2 * d_model if use_feature_identity else 0)
            self.skip_gate = nn.Sequential(nn.Linear(gate_in, d_model), nn.Sigmoid())
        if use_feature_identity:
            self.question_embed = nn.Embedding(num_features, d_model)
            self.group_embed = nn.Embedding(len(group_sizes), d_model)
            group_ids: list[int] = []
            for group_idx, size in enumerate(group_sizes):
                group_ids.extend([group_idx] * int(size))
            if len(group_ids) != num_features:
                raise ValueError(
                    f"group_sizes sum to {len(group_ids)} but num_features={num_features}",
                )
            self.register_buffer("question_ids", torch.arange(num_features, dtype=torch.long))
            self.register_buffer("group_ids", torch.tensor(group_ids, dtype=torch.long))
        else:
            self.question_embed = None
            self.group_embed = None

        self.register_buffer("norm_mean", torch.zeros(num_features))
        self.register_buffer("norm_std", torch.ones(num_features))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.norm_mean.copy_(mean)
        safe_std = std.clone()
        safe_std[safe_std == 0] = 1.0
        self.norm_std.copy_(safe_std)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, F) -> (B, F, d_model)"""
        normalized = (features - self.norm_mean) / self.norm_std
        raw_in = normalized.unsqueeze(-1)  # (B, F, 1)
        tokens = self.scalar_mlp(raw_in)
        if self.use_feature_identity:
            q_emb = self.question_embed(self.question_ids).unsqueeze(0)
            g_emb = self.group_embed(self.group_ids).unsqueeze(0)
            if self.identity_mode == "question_only":
                tokens = tokens + q_emb
            elif self.identity_mode == "group_only":
                tokens = tokens + g_emb
            else:
                tokens = tokens + q_emb + g_emb
        if self.skip_mode == "additive":
            tokens = tokens + self.raw_proj(raw_in)
        elif self.skip_mode == "gated":
            raw_val = self.raw_proj(raw_in)  # (B, F, d)
            gate_parts = [raw_val]
            if self.use_feature_identity:
                B = features.size(0)
                gate_parts.append(q_emb.expand(B, -1, -1))
                gate_parts.append(g_emb.expand(B, -1, -1))
            gate = self.skip_gate(torch.cat(gate_parts, dim=-1))
            tokens = tokens + gate * raw_val
        return tokens  # (B, F, d)


# ---------------------------------------------------------------------------
# Prediction heads (local copies — no import coupling to attn_agg)
# ---------------------------------------------------------------------------

class AidaItemOrdinalHead(nn.Module):
    def __init__(self, d_model: int, num_items: int) -> None:
        super().__init__()
        self.scalar_proj = nn.Linear(d_model, 1)
        nn.init.normal_(self.scalar_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.scalar_proj.bias)
        self.tau_base = nn.Parameter(torch.full((num_items,), -1.1))
        self.tau_deltas = nn.Parameter(torch.ones(num_items, _NUM_THRESHOLDS - 1))

    def forward(self, item_vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.scalar_proj(item_vectors).squeeze(-1)
        deltas = F.softplus(self.tau_deltas)
        tau_0 = self.tau_base
        tau_1 = tau_0 + deltas[:, 0]
        tau_2 = tau_1 + deltas[:, 1]
        tau = torch.stack([tau_0, tau_1, tau_2], dim=-1)
        cum_probs = torch.sigmoid(tau.unsqueeze(0) - s.unsqueeze(-1))
        zeros = torch.zeros(cum_probs.shape[0], cum_probs.shape[1], 1,
                            device=cum_probs.device, dtype=cum_probs.dtype)
        ones = torch.ones_like(zeros)
        padded = torch.cat([zeros, cum_probs, ones], dim=-1)
        class_probs = (padded[:, :, 1:] - padded[:, :, :-1]).clamp(min=1e-7)
        class_probs = class_probs / class_probs.sum(dim=-1, keepdim=True)
        levels = torch.arange(_NUM_LEVELS, device=item_vectors.device, dtype=item_vectors.dtype)
        expected = (class_probs * levels).sum(dim=-1)
        return expected, class_probs


class AidaItemSigmoidHead(nn.Module):
    def __init__(self, d_model: int, num_items: int) -> None:
        super().__init__()
        self.scalar_proj = nn.Linear(d_model, 1)
        nn.init.normal_(self.scalar_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.scalar_proj.bias)

    def forward(self, item_vectors: torch.Tensor) -> tuple[torch.Tensor, None]:
        logits = self.scalar_proj(item_vectors).squeeze(-1)
        return torch.sigmoid(logits) * _MAX_SCORE, None


# ---------------------------------------------------------------------------
# AidaAttnAgg
# ---------------------------------------------------------------------------

class AidaAttnAgg(nn.Module):
    """Item queries attend over 23 AIDA feature tokens."""

    def __init__(self, config: AidaAttnAggConfig) -> None:
        super().__init__()
        d = config.d_model
        self.tokenizer = AidaFeatureTokenizer(
            config.num_aida_features,
            config.aida_groups,
            d,
            config.scalar_mlp_hidden,
            use_feature_identity=config.use_feature_identity,
            skip_mode=getattr(config, "skip_mode", "none"),
            identity_mode=getattr(config, "identity_mode", "both"),
        )
        self.post_proj_norm = nn.LayerNorm(d)

        if config.num_encoder_layers > 0:
            enc = nn.TransformerEncoderLayer(
                d_model=d, nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc, config.num_encoder_layers)
        else:
            self.encoder = None

        self.item_queries = nn.Parameter(torch.randn(config.num_items, d) * 0.02)

        dec = nn.TransformerDecoderLayer(
            d_model=d, nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec, config.num_decoder_layers)

        if config.head_type == "sigmoid":
            self.head: AidaItemOrdinalHead | AidaItemSigmoidHead = AidaItemSigmoidHead(d, config.num_items)
        else:
            self.head = AidaItemOrdinalHead(d, config.num_items)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(B, 23) -> (B, num_items), (B, num_items, 4)|None"""
        tokens = self.encode_features(features)
        B = features.size(0)
        queries = self.item_queries.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(queries, tokens)
        return self.head(decoded)

    def encode_features(self, features: torch.Tensor) -> torch.Tensor:
        """(B, 23) -> (B, 23, d_model) encoded tokens."""
        tokens = self.tokenizer(features)
        tokens = self.post_proj_norm(tokens)
        if self.encoder is not None:
            tokens = self.encoder(tokens)
        return tokens


# ---------------------------------------------------------------------------
# AidaGRU
# ---------------------------------------------------------------------------

class AidaGRU(nn.Module):
    """AIDA-feature model with temporal GRU over prior sessions."""

    def __init__(self, config: AidaGRUConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        base_cfg = AidaAttnAggConfig(
            num_aida_features=config.num_aida_features,
            aida_groups=config.aida_groups,
            d_model=d, nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dropout=config.dropout,
            scalar_mlp_hidden=config.scalar_mlp_hidden,
            use_feature_identity=config.use_feature_identity,
            skip_mode=getattr(config, "skip_mode", "none"),
            num_items=config.num_items,
            head_type=config.head_type,
        )
        self.aggregator = AidaAttnAgg(base_cfg)

        self.history_gru = nn.GRU(
            input_size=d, hidden_size=config.gru_hidden_dim,
            num_layers=config.gru_num_layers,
            dropout=config.gru_dropout if config.gru_num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.lag_embedding = nn.Embedding(config.max_lag + 1, d, padding_idx=0)
        self.Wg = nn.Linear(config.gru_hidden_dim, d)
        self.Wh = nn.Linear(config.gru_hidden_dim, d)
        nn.init.zeros_(self.Wh.weight)
        nn.init.zeros_(self.Wh.bias)
        self.history_dropout_p = config.history_dropout

    def forward(
        self,
        cur_features: torch.Tensor,
        hist_features: torch.Tensor | None = None,
        hist_owner: torch.Tensor | None = None,
        hist_lag: torch.Tensor | None = None,
        hist_len: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B = cur_features.size(0)
        d = self.config.d_model
        J = self.config.num_items
        device = cur_features.device

        cur_tokens = self.aggregator.encode_features(cur_features)

        has_history = (
            hist_features is not None
            and hist_features.size(0) > 0
            and hist_len is not None
            and hist_len.sum() > 0
        )

        if has_history and self.training and self.history_dropout_p > 0:
            keep = torch.bernoulli(
                torch.full((B,), 1.0 - self.history_dropout_p, device=device),
            ).bool()
            if not keep.any():
                has_history = False
            elif not keep.all():
                mask_h = keep[hist_owner]
                hist_features = hist_features[mask_h]
                hist_lag = hist_lag[mask_h]
                old_owner = hist_owner[mask_h]
                new_len = torch.zeros(B, dtype=torch.long, device=device)
                for i in range(B):
                    new_len[i] = (old_owner == i).sum()
                hist_len = new_len
                hist_owner = old_owner
                has_history = hist_features.size(0) > 0

        if has_history:
            h = self._compute_history(hist_features, hist_owner, hist_lag, hist_len, B)
        else:
            h = torch.zeros(B, J, d, device=device, dtype=cur_features.dtype)

        queries = self.aggregator.item_queries.unsqueeze(0).expand(B, -1, -1)
        gate = torch.sigmoid(self.Wg(h))
        delta = self.Wh(h)
        if hist_len is not None:
            mask = (hist_len > 0).to(dtype=queries.dtype).view(B, 1, 1)
        else:
            mask = torch.zeros(B, 1, 1, device=device, dtype=queries.dtype)
        q_cond = queries + mask * gate * delta

        decoded = self.aggregator.decoder(q_cond, cur_tokens)
        return self.aggregator.head(decoded)

    def _compute_history(
        self,
        hist_features: torch.Tensor,
        hist_owner: torch.Tensor,
        hist_lag: torch.Tensor,
        hist_len: torch.Tensor,
        B: int,
    ) -> torch.Tensor:
        H = hist_features.size(0)
        J = self.config.num_items
        d = self.config.d_model
        device = hist_features.device

        hist_tokens = self.aggregator.encode_features(hist_features)
        queries = self.aggregator.item_queries.unsqueeze(0).expand(H, -1, -1)
        hist_summary = self.aggregator.decoder(queries, hist_tokens)  # (H, J, d)

        T_max = int(hist_len.max().item())
        dense = torch.zeros(B, T_max, J, d, device=device, dtype=hist_summary.dtype)

        positions = torch.zeros(H, dtype=torch.long, device=device)
        counters = torch.zeros(B, dtype=torch.long, device=device)
        for i in range(H):
            owner = hist_owner[i].item()
            positions[i] = counters[owner]
            counters[owner] += 1

        dense[hist_owner, positions] = hist_summary

        lag_dense = torch.zeros(B, T_max, dtype=torch.long, device=device)
        lag_dense[hist_owner, positions] = hist_lag.clamp(0, self.config.max_lag)
        lag_emb = self.lag_embedding(lag_dense)
        dense = dense + lag_emb.unsqueeze(2)

        gru_in = dense.permute(0, 2, 1, 3).reshape(B * J, T_max, d)
        gru_lengths = hist_len.repeat_interleave(J).cpu().clamp(min=1)
        packed = pack_padded_sequence(gru_in, gru_lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.history_gru(packed)
        h = h_n[-1].view(B, J, -1)

        no_hist = (hist_len == 0).unsqueeze(-1).unsqueeze(-1)
        return h.masked_fill(no_hist, 0.0)


# ---------------------------------------------------------------------------
# AidaStackAttn
# ---------------------------------------------------------------------------

class _AidaHistoryCrossLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        out, _ = self.cross_attn(query, memory, memory, need_weights=False)
        ctx = self.attn_norm(out)
        return self.ff_norm(ctx + self.ff(ctx))


class AidaStackAttn(nn.Module):
    """AIDA-feature model with cross-attention to previous session."""

    def __init__(self, config: AidaStackAttnConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        base_cfg = AidaAttnAggConfig(
            num_aida_features=config.num_aida_features,
            aida_groups=config.aida_groups,
            d_model=d, nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dropout=config.dropout,
            scalar_mlp_hidden=config.scalar_mlp_hidden,
            use_feature_identity=config.use_feature_identity,
            skip_mode=getattr(config, "skip_mode", "none"),
            num_items=config.num_items,
            head_type=config.head_type,
        )
        self.aggregator = AidaAttnAgg(base_cfg)

        self.history_cross_attn = nn.MultiheadAttention(d, config.nhead, dropout=config.dropout, batch_first=True)
        self.history_attn_norm = nn.LayerNorm(d)
        self.history_ff = nn.Sequential(
            nn.Linear(d, config.dim_feedforward), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, d), nn.Dropout(config.dropout),
        )
        self.history_ff_norm = nn.LayerNorm(d)
        self.history_gate = nn.Linear(d * 2, 1)
        self.history_out = nn.Linear(d, d)
        nn.init.zeros_(self.history_out.weight)
        nn.init.zeros_(self.history_out.bias)

        self.history_extra_layers = nn.ModuleList([
            _AidaHistoryCrossLayer(d, config.nhead, config.dim_feedforward, config.dropout)
            for _ in range(max(0, config.history_num_layers - 1))
        ])
        self.history_dropout_p = config.history_dropout

    def forward(
        self,
        cur_features: torch.Tensor,
        prev_features: torch.Tensor | None = None,
        has_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B = cur_features.size(0)
        cur_tokens = self.aggregator.encode_features(cur_features)
        queries = self.aggregator.item_queries.unsqueeze(0).expand(B, -1, -1)
        cur_items = self.aggregator.decoder(queries, cur_tokens)

        final_items = cur_items
        use_prev = (
            prev_features is not None
            and has_prev is not None
            and bool(has_prev.any().item())
        )

        if use_prev and self.training and self.history_dropout_p > 0:
            keep = torch.bernoulli(
                torch.full((B,), 1.0 - self.history_dropout_p, device=cur_features.device),
            ).bool()
            has_prev = has_prev & keep
            use_prev = bool(has_prev.any().item())

        if use_prev:
            idx = torch.nonzero(has_prev, as_tuple=False).squeeze(-1)
            prev_tokens = self.aggregator.encode_features(prev_features[idx])
            cur_subset = cur_items[idx]

            hist_ctx, _ = self.history_cross_attn(cur_subset, prev_tokens, prev_tokens, need_weights=False)
            hist_ctx = self.history_attn_norm(hist_ctx)
            hist_features_out = self.history_ff_norm(hist_ctx + self.history_ff(hist_ctx))

            for layer in self.history_extra_layers:
                hist_features_out = layer(hist_features_out, prev_tokens)

            gate = torch.sigmoid(self.history_gate(torch.cat([cur_subset, hist_features_out], dim=-1)))
            delta = self.history_out(hist_features_out)
            final_items = cur_items.clone()
            final_items[idx] = cur_subset + gate * delta
        else:
            # Keep trainable params in the graph when aggregator is frozen
            final_items = cur_items + 0.0 * self.history_out(torch.zeros_like(cur_items))

        return self.aggregator.head(final_items)


# ---------------------------------------------------------------------------
# AidaEmbedAttnAgg — hybrid: AIDA tokens + pooled sentence summary
# ---------------------------------------------------------------------------

_EMBED_ATTNAGG_MODEL = "aida_embed_attnagg_model.pt"
_EMBED_ATTNAGG_CONFIG = "aida_embed_attnagg_config.json"


class AidaEmbedAttnAgg(nn.Module):
    """Hybrid: AIDA feature tokens + one pooled sentence-embedding summary token."""

    def __init__(self, config: AidaEmbedAttnAggConfig) -> None:
        super().__init__()
        d = config.d_model
        self.config = config

        self.tokenizer = AidaFeatureTokenizer(
            config.num_aida_features,
            config.aida_groups,
            d,
            config.scalar_mlp_hidden,
            use_feature_identity=config.use_feature_identity,
        )
        self.sent_proj = nn.Linear(config.backbone_embed_dim, d)
        self.post_proj_norm = nn.LayerNorm(d)

        if config.num_encoder_layers > 0:
            enc = nn.TransformerEncoderLayer(
                d_model=d, nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc, config.num_encoder_layers)
        else:
            self.encoder = None

        self.item_queries = nn.Parameter(torch.randn(config.num_items, d) * 0.02)

        dec = nn.TransformerDecoderLayer(
            d_model=d, nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec, config.num_decoder_layers)

        if config.head_type == "sigmoid":
            self.head: AidaItemOrdinalHead | AidaItemSigmoidHead = AidaItemSigmoidHead(d, config.num_items)
        else:
            self.head = AidaItemOrdinalHead(d, config.num_items)

    def forward(
        self,
        features: torch.Tensor,
        sent_embeddings: torch.Tensor,
        sent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            features: (B, F) AIDA scalar features
            sent_embeddings: (B, N, embed_dim) sentence embeddings
            sent_mask: (B, N) 1.0 for real tokens, 0.0 for padding
        """
        # AIDA tokens: (B, F, d)
        aida_tokens = self.tokenizer(features)

        # Pool sentence embeddings -> (B, 1, d)
        mask_sum = sent_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        pooled = (sent_embeddings * sent_mask.unsqueeze(-1)).sum(dim=1) / mask_sum  # (B, embed_dim)
        sent_token = self.sent_proj(pooled).unsqueeze(1)  # (B, 1, d)

        # Concatenate: (B, F+1, d)
        tokens = torch.cat([aida_tokens, sent_token], dim=1)
        tokens = self.post_proj_norm(tokens)
        if self.encoder is not None:
            tokens = self.encoder(tokens)

        B = features.size(0)
        queries = self.item_queries.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(queries, tokens)
        return self.head(decoded)


def save_checkpoint_embed_attnagg(
    model: AidaEmbedAttnAgg, path: str | Path, config: AidaEmbedAttnAggConfig,
) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / _EMBED_ATTNAGG_MODEL)
    d = {
        "model_type": "aida_embed_attnagg",
        "num_aida_features": config.num_aida_features,
        "aida_groups": config.aida_groups,
        "d_model": config.d_model,
        "nhead": config.nhead,
        "dim_feedforward": config.dim_feedforward,
        "num_encoder_layers": config.num_encoder_layers,
        "num_decoder_layers": config.num_decoder_layers,
        "dropout": config.dropout,
        "scalar_mlp_hidden": config.scalar_mlp_hidden,
        "use_feature_identity": config.use_feature_identity,
        "backbone_embed_dim": config.backbone_embed_dim,
        "max_sentences": config.max_sentences,
        "num_items": config.num_items,
        "head_type": config.head_type,
        "item_loss_weight": config.item_loss_weight,
        "aida_cache_path": config.aida_cache_path,
        "aida_extra_cache_path": config.aida_extra_cache_path,
        "embed_cache_dir": config.embed_cache_dir,
        "result_name": config.result_name,
    }
    (path / _EMBED_ATTNAGG_CONFIG).write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    logger.info("AidaEmbedAttnAgg checkpoint saved to %s", path)
    return path


def load_checkpoint_embed_attnagg(path: str | Path) -> AidaEmbedAttnAgg:
    path = Path(path)
    d = json.loads((path / _EMBED_ATTNAGG_CONFIG).read_text(encoding="utf-8"))
    cfg = AidaEmbedAttnAggConfig(
        num_aida_features=d["num_aida_features"],
        aida_groups=d["aida_groups"],
        d_model=d["d_model"],
        nhead=d["nhead"],
        dim_feedforward=d["dim_feedforward"],
        num_encoder_layers=d["num_encoder_layers"],
        num_decoder_layers=d["num_decoder_layers"],
        dropout=d["dropout"],
        scalar_mlp_hidden=d["scalar_mlp_hidden"],
        use_feature_identity=d.get("use_feature_identity", True),
        backbone_embed_dim=d.get("backbone_embed_dim", 4096),
        max_sentences=d.get("max_sentences", 384),
        num_items=d["num_items"],
        head_type=d.get("head_type", "sigmoid"),
        item_loss_weight=d["item_loss_weight"],
    )
    model = AidaEmbedAttnAgg(cfg)
    state = torch.load(path / _EMBED_ATTNAGG_MODEL, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def predict_from_cache_embed_attnagg(
    model: AidaEmbedAttnAgg,
    aida_cache: dict[str, list[float]],
    embed_cache_dir: Path,
    run_ids_and_sessions: list[tuple[str, int]],
    max_sentences: int = 384,
) -> list[dict[str, float]]:
    from EmoTrack.generative.schema import PHQ8_SYMPTOMS as _PHQ8_SYMPTOMS

    model.eval()
    device = next(model.parameters()).device
    results: list[dict[str, float]] = []

    with torch.no_grad():
        for run_id, session_index in run_ids_and_sessions:
            # AIDA features
            key = f"{run_id}::s{session_index}"
            if key not in aida_cache:
                raise KeyError(f"AIDA features not in cache: {key}")
            aida_feat = torch.tensor([aida_cache[key]], dtype=torch.float32, device=device)

            # Sentence embeddings
            is_daic = run_id.isdigit()
            subdir = "daic" if is_daic else "longcounsel8"
            emb_path = embed_cache_dir / subdir / run_id / f"session_{session_index}.pt"
            if not emb_path.exists():
                raise FileNotFoundError(f"Embedding cache not found: {emb_path}")
            cached = torch.load(emb_path, map_location="cpu", weights_only=False)
            emb = cached["client_embeddings"]
            n = min(emb.size(0), max_sentences)
            emb = emb[:n]

            sent = torch.zeros(1, max_sentences, emb.size(1), device=device)
            sent[0, :n] = emb.to(device)
            mask = torch.zeros(1, max_sentences, device=device)
            mask[0, :n] = 1.0

            expected, _ = model(aida_feat, sent, mask)
            scores = expected.squeeze(0).float().cpu().tolist()
            results.append({s: round(v, 2) for s, v in zip(_PHQ8_SYMPTOMS, scores)})

    return results


# ---------------------------------------------------------------------------
# AidaEmbedHybrid — full sentence tokens + AIDA tokens
# ---------------------------------------------------------------------------

_HYBRID_MODEL = "aida_embed_hybrid_model.pt"
_HYBRID_CONFIG = "aida_embed_hybrid_config.json"

_PREV_SENT_MODES = ("none", "summary_only", "summary_retrieval")
_INPUT_MODES = ("full", "aida_only", "embedding_only")
_READOUT_MODES = ("item_queries", "pooled")


class AidaEmbedHybrid(nn.Module):
    """Hybrid model using full sentence token sequence + AIDA feature tokens.

    Production fusion mode: full_seq — concat all sentence + AIDA tokens, single decoder.
    Optional prev_sent_mode over previous-session sentence embeddings:
      - none: no previous-session memory path.
      - summary_only: gated mean-pool summary update.
      - summary_retrieval: summary update plus compressed-slot retrieval.
    """

    def __init__(self, config) -> None:
        super().__init__()
        d = config.d_model
        self.config = config
        self.speaker_view = getattr(config, "speaker_view", "client")
        self.psyche_transcript_root = getattr(config, "psyche_transcript_root", "")
        self.daic_transcript_root = getattr(config, "daic_transcript_root", "")
        self.input_mode = getattr(config, "input_mode", "full")
        self.readout_mode = getattr(config, "readout_mode", "item_queries")
        if self.input_mode not in _INPUT_MODES:
            raise ValueError(f"input_mode must be one of {_INPUT_MODES}, got {self.input_mode!r}")
        if self.readout_mode not in _READOUT_MODES:
            raise ValueError(f"readout_mode must be one of {_READOUT_MODES}, got {self.readout_mode!r}")
        if self.input_mode == "aida_only" and config.prev_sent_mode != "none":
            raise ValueError("input_mode='aida_only' requires prev_sent_mode='none'")
        if self.readout_mode == "pooled" and config.prev_sent_mode != "none":
            raise ValueError("readout_mode='pooled' requires prev_sent_mode='none'")

        # AIDA branch
        self.tokenizer = AidaFeatureTokenizer(
            config.num_aida_features, config.aida_groups, d,
            config.scalar_mlp_hidden, use_feature_identity=config.use_feature_identity,
            skip_mode=config.skip_mode,
            identity_mode=config.identity_mode,
        )

        # Sentence branch
        self.sent_proj = nn.Linear(config.backbone_embed_dim, d)
        self.sent_norm = nn.LayerNorm(d)

        # Shared components
        self.post_proj_norm = nn.LayerNorm(d)
        if self.readout_mode == "item_queries":
            self.item_queries = nn.Parameter(torch.randn(config.num_items, d) * 0.02)
            layer = nn.TransformerDecoderLayer(
                d_model=d, nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(layer, config.num_decoder_layers)
            self.pooled_to_items = None
        else:
            self.item_queries = None
            self.decoder = None
            self.pooled_to_items = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, config.num_items * d),
            )

        if config.num_encoder_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, config.num_encoder_layers)
        else:
            self.encoder = None

        if config.head_type == "sigmoid":
            self.head: AidaItemOrdinalHead | AidaItemSigmoidHead = AidaItemSigmoidHead(d, config.num_items)
        else:
            self.head = AidaItemOrdinalHead(d, config.num_items)

        # Previous-session sentence memory
        if config.prev_sent_mode not in _PREV_SENT_MODES:
            raise ValueError(
                f"prev_sent_mode must be one of {_PREV_SENT_MODES}, "
                f"got {config.prev_sent_mode!r}",
            )
        self.prev_sent_mode = config.prev_sent_mode
        if self.prev_sent_mode != "none":
            self.prev_sent_proj = nn.Linear(config.backbone_embed_dim, d)
            self.prev_sent_norm = nn.LayerNorm(d)
            self.ps_summary_proj = nn.Linear(d, d)
            self.ps_summary_gate = nn.Linear(d * 2, 1)
        if self.prev_sent_mode == "summary_retrieval":
            ns = config.ps_num_slots
            self.ps_slot_queries = nn.Parameter(torch.randn(ns, d) * 0.02)
            self.ps_compress = nn.MultiheadAttention(d, config.nhead, dropout=config.dropout, batch_first=True)
            self.ps_compress_norm = nn.LayerNorm(d)
            self.ps_item_cross = nn.MultiheadAttention(d, config.nhead, dropout=config.dropout, batch_first=True)
            self.ps_item_cross_norm = nn.LayerNorm(d)
            self.ps_item_ff = nn.Sequential(
                nn.Linear(d, config.dim_feedforward), nn.GELU(), nn.Dropout(config.dropout),
                nn.Linear(config.dim_feedforward, d), nn.Dropout(config.dropout),
            )
            self.ps_item_ff_norm = nn.LayerNorm(d)
            self.ps_gate = nn.Linear(d * 2, 1)
            self.ps_out = nn.Linear(d, d)
            nn.init.zeros_(self.ps_out.weight)
            nn.init.zeros_(self.ps_out.bias)
        self.prev_sent_dropout_p = config.history_dropout

    def _apply_prev_sent(
        self,
        decoded: torch.Tensor,
        prev_sent_tokens: torch.Tensor,
        prev_sent_pad: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply previous-session sentence memory to decoded item vectors."""
        B = decoded.size(0)
        has_prev_rows = (
            torch.ones(B, dtype=torch.bool, device=decoded.device)
            if prev_sent_pad is None
            else ~prev_sent_pad.all(dim=1)
        )
        if not has_prev_rows.any():
            return decoded

        # Training-time dropout of prev-sent memory
        if self.training and self.prev_sent_dropout_p > 0:
            keep = torch.bernoulli(
                torch.full((B,), 1.0 - self.prev_sent_dropout_p, device=decoded.device),
            ).bool()
            has_prev_rows = has_prev_rows & keep
            if not keep.any():
                return decoded
            if not has_prev_rows.any():
                return decoded

        decoded_with_prev = decoded[has_prev_rows]
        prev_sent_tokens = prev_sent_tokens[has_prev_rows]
        if prev_sent_pad is not None:
            prev_sent_pad = prev_sent_pad[has_prev_rows]

        if self.prev_sent_mode not in ("summary_only", "summary_retrieval"):
            return decoded

        valid = (
            (~prev_sent_pad).float().unsqueeze(-1)
            if prev_sent_pad is not None
            else torch.ones_like(prev_sent_tokens[:, :, :1])
        )
        summary = (prev_sent_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        summary = self.ps_summary_proj(summary).unsqueeze(1).expand_as(decoded_with_prev)

        if self.prev_sent_mode == "summary_retrieval":
            slots = self.ps_slot_queries.unsqueeze(0).expand(decoded_with_prev.size(0), -1, -1)
            mem, _ = self.ps_compress(
                slots,
                prev_sent_tokens,
                prev_sent_tokens,
                key_padding_mask=prev_sent_pad,
                need_weights=False,
            )
            mem = self.ps_compress_norm(mem)

            ctx, _ = self.ps_item_cross(decoded_with_prev, mem, mem, need_weights=False)
            ctx = self.ps_item_cross_norm(ctx)
            ctx = self.ps_item_ff_norm(ctx + self.ps_item_ff(ctx))
            # Gated retrieval update
            gate = torch.sigmoid(self.ps_gate(torch.cat([decoded_with_prev, ctx], dim=-1)))
            delta = self.ps_out(ctx)
            decoded_with_prev = decoded_with_prev + gate * delta

        sg = torch.sigmoid(
            self.ps_summary_gate(torch.cat([decoded_with_prev, summary], dim=-1))
        )
        updated = decoded_with_prev + sg * summary

        decoded = decoded.clone()
        decoded[has_prev_rows] = updated
        return decoded

    def forward(
        self,
        features: torch.Tensor,
        sent_embeddings: torch.Tensor,
        sent_mask: torch.Tensor,
        prev_sent_embeddings: torch.Tensor | None = None,
        prev_sent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B = features.size(0)
        tokens, pad_mask = self._encode_current_tokens(features, sent_embeddings, sent_mask)

        prev_sent_tokens = None
        prev_sent_pad = None
        has_prev_sent = (
            prev_sent_embeddings is not None
            and (prev_sent_mask is None or bool(prev_sent_mask.bool().any().item()))
        )
        if self.prev_sent_mode != "none" and has_prev_sent:
            prev_sent_tokens = self.prev_sent_proj(prev_sent_embeddings)
            prev_sent_tokens = self.prev_sent_norm(prev_sent_tokens)
            prev_sent_pad = (1.0 - prev_sent_mask).bool() if prev_sent_mask is not None else None

        if self.encoder is not None:
            tokens = self.encoder(tokens, src_key_padding_mask=pad_mask)

        if self.readout_mode == "pooled":
            item_vectors = self._pooled_item_vectors(tokens, pad_mask)
            return self.head(item_vectors)

        queries = self.item_queries.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(queries, tokens, memory_key_padding_mask=pad_mask)

        if self.prev_sent_mode != "none" and prev_sent_tokens is not None:
            decoded = self._apply_prev_sent(decoded, prev_sent_tokens, prev_sent_pad)

        return self.head(decoded)

    def _encode_current_tokens(
        self,
        features: torch.Tensor,
        sent_embeddings: torch.Tensor,
        sent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = features.size(0)
        token_parts: list[torch.Tensor] = []
        pad_parts: list[torch.Tensor] = []

        if self.input_mode in ("full", "aida_only"):
            aida_tokens = self.tokenizer(features)
            aida_tokens = self.post_proj_norm(aida_tokens)
            token_parts.append(aida_tokens)
            pad_parts.append(torch.zeros(B, aida_tokens.size(1), device=features.device, dtype=torch.bool))

        if self.input_mode in ("full", "embedding_only"):
            sent_tokens = self.sent_proj(sent_embeddings)
            sent_tokens = self.sent_norm(sent_tokens)
            token_parts.append(sent_tokens)
            pad_parts.append((1.0 - sent_mask).bool())

        return torch.cat(token_parts, dim=1), torch.cat(pad_parts, dim=1)

    def _pooled_item_vectors(self, tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        valid = (~pad_mask).to(dtype=tokens.dtype).unsqueeze(-1)
        pooled = (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.pooled_to_items(pooled).view(tokens.size(0), self.config.num_items, self.config.d_model)


def save_checkpoint_hybrid(model: AidaEmbedHybrid, path: str | Path, config) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / _HYBRID_MODEL)
    d = {
        "model_type": "aida_embed_hybrid",
        "fusion_mode": "full_seq",
        "num_aida_features": config.num_aida_features,
        "aida_groups": config.aida_groups,
        "d_model": config.d_model,
        "nhead": config.nhead,
        "dim_feedforward": config.dim_feedforward,
        "num_encoder_layers": config.num_encoder_layers,
        "num_decoder_layers": config.num_decoder_layers,
        "dropout": config.dropout,
        "scalar_mlp_hidden": config.scalar_mlp_hidden,
        "use_feature_identity": config.use_feature_identity,
        "backbone_embed_dim": config.backbone_embed_dim,
        "max_sentences": config.max_sentences,
        "num_items": config.num_items,
        "head_type": config.head_type,
        "item_loss_weight": config.item_loss_weight,
        "fn_loss_weight": config.fn_loss_weight,
        "fn_threshold": config.fn_threshold,
        "aida_cache_path": config.aida_cache_path,
        "aida_extra_cache_path": config.aida_extra_cache_path,
        "embed_cache_dir": config.embed_cache_dir,
        "result_name": config.result_name,
        "seed": getattr(config, "seed", None),
        "inference_seed": getattr(config, "inference_seed", None),
        "deterministic_training": getattr(config, "deterministic_training", False),
        "skip_mode": config.skip_mode,
        "identity_mode": config.identity_mode,
        "input_mode": config.input_mode,
        "readout_mode": config.readout_mode,
        "history_dropout": config.history_dropout,
        "prev_sent_mode": config.prev_sent_mode,
        "ps_num_slots": config.ps_num_slots,
        "speaker_view": getattr(config, "speaker_view", "client"),
        "psyche_transcript_root": getattr(config, "psyche_transcript_root", ""),
        "daic_transcript_root": getattr(config, "daic_transcript_root", ""),
    }
    (path / _HYBRID_CONFIG).write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    logger.info("AidaEmbedHybrid checkpoint saved to %s", path)
    return path


def load_checkpoint_hybrid(path: str | Path) -> AidaEmbedHybrid:
    from EmoTrack.train.aida.config import AidaEmbedHybridConfig
    path = Path(path)
    d = json.loads((path / _HYBRID_CONFIG).read_text(encoding="utf-8"))
    stored_fusion = d.get("fusion_mode", "full_seq")
    if stored_fusion != "full_seq":
        raise ValueError(
            f"Checkpoint at {path} was trained with fusion_mode={stored_fusion!r}, "
            f"which has been removed. Only 'full_seq' checkpoints can be loaded."
        )
    cfg = AidaEmbedHybridConfig(
        num_aida_features=d["num_aida_features"],
        aida_groups=d["aida_groups"],
        d_model=d["d_model"],
        nhead=d["nhead"],
        dim_feedforward=d["dim_feedforward"],
        num_encoder_layers=d["num_encoder_layers"],
        num_decoder_layers=d["num_decoder_layers"],
        dropout=d["dropout"],
        scalar_mlp_hidden=d["scalar_mlp_hidden"],
        use_feature_identity=d.get("use_feature_identity", True),
        backbone_embed_dim=d.get("backbone_embed_dim", 4096),
        max_sentences=d.get("max_sentences", 384),
        num_items=d["num_items"],
        head_type=d.get("head_type", "sigmoid"),
        skip_mode=d.get("skip_mode", "none"),
        identity_mode=d.get("identity_mode", "both"),
        input_mode=d["input_mode"],
        readout_mode=d["readout_mode"],
        item_loss_weight=d["item_loss_weight"],
        fn_loss_weight=d.get("fn_loss_weight", 0.0),
        fn_threshold=d.get("fn_threshold", 10.0),
        history_dropout=d.get("history_dropout", 0.2),
        prev_sent_mode=d.get("prev_sent_mode", "none"),
        ps_num_slots=d.get("ps_num_slots", 16),
        speaker_view=d.get("speaker_view", "client"),
        psyche_transcript_root=d.get("psyche_transcript_root", ""),
        daic_transcript_root=d.get("daic_transcript_root", ""),
    )
    model = AidaEmbedHybrid(cfg)
    state = torch.load(path / _HYBRID_MODEL, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def predict_from_cache_hybrid(
    model: AidaEmbedHybrid,
    aida_cache: dict[str, list[float]],
    embed_cache_dir: Path,
    run_ids_and_sessions: list[tuple[str, int]],
    max_sentences: int | None = None,
) -> list[dict[str, float]]:
    from EmoTrack.generative.schema import PHQ8_SYMPTOMS as _PHQ8_SYMPTOMS
    from EmoTrack.train.aida.dataset import _select_session_embeddings

    model.eval()
    device = next(model.parameters()).device
    results: list[dict[str, float]] = []
    speaker_view = getattr(model, "speaker_view", "client")
    psyche_transcript_root = getattr(model, "psyche_transcript_root", "")
    daic_transcript_root = getattr(model, "daic_transcript_root", "")
    input_mode = getattr(model, "input_mode", "full")
    if max_sentences is None:
        max_sentences = model.config.max_sentences

    def transcript_path(run_id: str) -> Path | None:
        if speaker_view != "ordered":
            return None
        is_daic_run = run_id.isdigit()
        root = daic_transcript_root if is_daic_run else psyche_transcript_root
        if not root:
            raise ValueError(
                "speaker_view='ordered' checkpoint is missing the transcript root",
            )
        return Path(root) / run_id / "common" / "transcript.jsonl"

    with torch.no_grad():
        for run_id, session_index in run_ids_and_sessions:
            key = f"{run_id}::s{session_index}"
            if key not in aida_cache:
                raise KeyError(f"AIDA features not in cache: {key}")
            aida_feat = torch.tensor([aida_cache[key]], dtype=torch.float32, device=device)

            is_daic = run_id.isdigit()
            subdir = "daic" if is_daic else "longcounsel8"
            if input_mode == "aida_only":
                sent = torch.zeros(1, 0, model.config.backbone_embed_dim, device=device)
                mask = torch.zeros(1, 0, device=device)
            else:
                emb_path = embed_cache_dir / subdir / run_id / f"session_{session_index}.pt"
                if not emb_path.exists():
                    raise FileNotFoundError(f"Embedding cache not found: {emb_path}")
                cached = torch.load(emb_path, map_location="cpu", weights_only=False)
                emb = _select_session_embeddings(
                    cached,
                    speaker_view,
                    transcript_path(run_id),
                    session_index,
                )
                n = min(emb.size(0), max_sentences)
                emb = emb[:n]

                sent = torch.zeros(1, max_sentences, emb.size(1), device=device)
                sent[0, :n] = emb.to(device)
                mask = torch.zeros(1, max_sentences, device=device)
                mask[0, :n] = 1.0

            fwd_kwargs: dict = {}
            if getattr(model, "prev_sent_mode", "none") != "none":
                prev_sent_t = None
                prev_sent_mask_t = None
                if not is_daic and session_index > 1:
                    prev_emb_path = embed_cache_dir / subdir / run_id / f"session_{session_index - 1}.pt"
                    if prev_emb_path.exists():
                        prev_cached = torch.load(prev_emb_path, map_location="cpu", weights_only=False)
                        prev_emb = _select_session_embeddings(
                            prev_cached,
                            speaker_view,
                            transcript_path(run_id),
                            session_index - 1,
                        )
                        pn = min(prev_emb.size(0), max_sentences)
                        prev_sent_t = torch.zeros(1, max_sentences, prev_emb.size(1), device=device)
                        prev_sent_t[0, :pn] = prev_emb[:pn].to(device)
                        prev_sent_mask_t = torch.zeros(1, max_sentences, device=device)
                        prev_sent_mask_t[0, :pn] = 1.0
                fwd_kwargs["prev_sent_embeddings"] = prev_sent_t
                fwd_kwargs["prev_sent_mask"] = prev_sent_mask_t
            expected, _ = model(aida_feat, sent, mask, **fwd_kwargs)
            scores = expected.squeeze(0).float().cpu().tolist()
            results.append({s: round(v, 2) for s, v in zip(_PHQ8_SYMPTOMS, scores)})

    return results


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def _base_config_dict(config: AidaAttnAggConfig) -> dict:
    return {
        "num_aida_features": config.num_aida_features,
        "aida_groups": config.aida_groups,
        "d_model": config.d_model,
        "nhead": config.nhead,
        "dim_feedforward": config.dim_feedforward,
        "num_encoder_layers": config.num_encoder_layers,
        "num_decoder_layers": config.num_decoder_layers,
        "dropout": config.dropout,
        "scalar_mlp_hidden": config.scalar_mlp_hidden,
        "use_feature_identity": config.use_feature_identity,
        "skip_mode": getattr(config, "skip_mode", "none"),
        "identity_mode": getattr(config, "identity_mode", "both"),
        "num_items": config.num_items,
        "head_type": config.head_type,
        "item_loss_weight": config.item_loss_weight,
        "aida_cache_path": config.aida_cache_path,
        "aida_extra_cache_path": config.aida_extra_cache_path,
        "result_name": config.result_name,
    }


def save_checkpoint_attnagg(model: AidaAttnAgg, path: str | Path, config: AidaAttnAggConfig) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / _ATTNAGG_MODEL)
    d = {"model_type": "aida_attnagg", **_base_config_dict(config)}
    (path / _ATTNAGG_CONFIG).write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    logger.info("AidaAttnAgg checkpoint saved to %s", path)
    return path


def save_checkpoint_gru(model: AidaGRU, path: str | Path, config: AidaGRUConfig) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / _GRU_MODEL)
    d = {
        "model_type": "aida_gru",
        **_base_config_dict(config),
        "gru_hidden_dim": config.gru_hidden_dim,
        "gru_num_layers": config.gru_num_layers,
        "gru_dropout": config.gru_dropout,
        "history_dropout": config.history_dropout,
        "max_lag": config.max_lag,
    }
    (path / _GRU_CONFIG).write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    logger.info("AidaGRU checkpoint saved to %s", path)
    return path


def save_checkpoint_stackattn(model: AidaStackAttn, path: str | Path, config: AidaStackAttnConfig) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / _STACKATTN_MODEL)
    d = {
        "model_type": "aida_stackattn",
        **_base_config_dict(config),
        "history_num_layers": config.history_num_layers,
        "history_dropout": config.history_dropout,
    }
    (path / _STACKATTN_CONFIG).write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    logger.info("AidaStackAttn checkpoint saved to %s", path)
    return path


def _build_attnagg_from_dict(d: dict) -> AidaAttnAgg:
    cfg = AidaAttnAggConfig(
        num_aida_features=d["num_aida_features"],
        aida_groups=d["aida_groups"],
        d_model=d["d_model"],
        nhead=d["nhead"],
        dim_feedforward=d["dim_feedforward"],
        num_encoder_layers=d["num_encoder_layers"],
        num_decoder_layers=d["num_decoder_layers"],
        dropout=d["dropout"],
        scalar_mlp_hidden=d["scalar_mlp_hidden"],
        use_feature_identity=d.get("use_feature_identity", False),
        skip_mode=d.get("skip_mode", "none"),
        identity_mode=d.get("identity_mode", "both"),
        num_items=d["num_items"],
        head_type=d.get("head_type", "sigmoid"),
        item_loss_weight=d["item_loss_weight"],
    )
    return AidaAttnAgg(cfg)


def load_checkpoint_attnagg(path: str | Path) -> AidaAttnAgg:
    path = Path(path)
    d = json.loads((path / _ATTNAGG_CONFIG).read_text(encoding="utf-8"))
    model = _build_attnagg_from_dict(d)
    state = torch.load(path / _ATTNAGG_MODEL, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def load_checkpoint_gru(path: str | Path) -> AidaGRU:
    path = Path(path)
    d = json.loads((path / _GRU_CONFIG).read_text(encoding="utf-8"))
    cfg = AidaGRUConfig(
        num_aida_features=d["num_aida_features"],
        aida_groups=d["aida_groups"],
        d_model=d["d_model"],
        nhead=d["nhead"],
        dim_feedforward=d["dim_feedforward"],
        num_encoder_layers=d["num_encoder_layers"],
        num_decoder_layers=d["num_decoder_layers"],
        dropout=d["dropout"],
        scalar_mlp_hidden=d["scalar_mlp_hidden"],
        use_feature_identity=d.get("use_feature_identity", False),
        num_items=d["num_items"],
        head_type=d.get("head_type", "sigmoid"),
        item_loss_weight=d["item_loss_weight"],
        gru_hidden_dim=d["gru_hidden_dim"],
        gru_num_layers=d["gru_num_layers"],
        gru_dropout=d.get("gru_dropout", 0.0),
        history_dropout=d.get("history_dropout", 0.0),
        max_lag=d.get("max_lag", 10),
    )
    model = AidaGRU(cfg)
    state = torch.load(path / _GRU_MODEL, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def load_checkpoint_stackattn(path: str | Path) -> AidaStackAttn:
    path = Path(path)
    d = json.loads((path / _STACKATTN_CONFIG).read_text(encoding="utf-8"))
    cfg = AidaStackAttnConfig(
        num_aida_features=d["num_aida_features"],
        aida_groups=d["aida_groups"],
        d_model=d["d_model"],
        nhead=d["nhead"],
        dim_feedforward=d["dim_feedforward"],
        num_encoder_layers=d["num_encoder_layers"],
        num_decoder_layers=d["num_decoder_layers"],
        dropout=d["dropout"],
        scalar_mlp_hidden=d["scalar_mlp_hidden"],
        use_feature_identity=d.get("use_feature_identity", False),
        skip_mode=d.get("skip_mode", "none"),
        identity_mode=d.get("identity_mode", "both"),
        num_items=d["num_items"],
        head_type=d.get("head_type", "sigmoid"),
        item_loss_weight=d["item_loss_weight"],
        history_num_layers=d.get("history_num_layers", 1),
        history_dropout=d.get("history_dropout", 0.0),
    )
    model = AidaStackAttn(cfg)
    state = torch.load(path / _STACKATTN_MODEL, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


# ---------------------------------------------------------------------------
# predict_from_cache
# ---------------------------------------------------------------------------

def predict_from_cache_attnagg(
    model: AidaAttnAgg,
    aida_cache: dict[str, list[float]],
    run_ids_and_sessions: list[tuple[str, int]],
) -> list[dict[str, float]]:
    from EmoTrack.generative.schema import PHQ8_SYMPTOMS as _PHQ8_SYMPTOMS

    model.eval()
    device = next(model.parameters()).device
    features_list = []
    for run_id, session_index in run_ids_and_sessions:
        key = f"{run_id}::s{session_index}"
        if key not in aida_cache:
            raise KeyError(f"AIDA features not in cache: {key}")
        features_list.append(aida_cache[key])

    results: list[dict[str, float]] = []
    with torch.no_grad():
        batch = torch.tensor(features_list, dtype=torch.float32, device=device)
        expected, _ = model(batch)
        for scores in expected.float().cpu().tolist():
            results.append({s: round(v, 2) for s, v in zip(_PHQ8_SYMPTOMS, scores)})
    return results


def predict_from_cache_gru(
    model: AidaGRU,
    aida_cache: dict[str, list[float]],
    run_ids_and_sessions: list[tuple[str, int]],
) -> list[dict[str, float]]:
    from EmoTrack.generative.schema import PHQ8_SYMPTOMS as _PHQ8_SYMPTOMS

    model.eval()
    device = next(model.parameters()).device
    results: list[dict[str, float]] = []

    with torch.no_grad():
        for run_id, session_index in run_ids_and_sessions:
            key = f"{run_id}::s{session_index}"
            if key not in aida_cache:
                raise KeyError(f"AIDA features not in cache: {key}")
            cur = torch.tensor([aida_cache[key]], dtype=torch.float32, device=device)

            hist_feats, hist_lags = [], []
            is_daic = run_id.isdigit()
            if not is_daic:
                for prior in range(1, session_index):
                    pk = f"{run_id}::s{prior}"
                    if pk in aida_cache:
                        hist_feats.append(aida_cache[pk])
                        hist_lags.append(session_index - prior)

            if hist_feats:
                H = len(hist_feats)
                hf = torch.tensor(hist_feats, dtype=torch.float32, device=device)
                ho = torch.zeros(H, dtype=torch.long, device=device)
                hl = torch.tensor(hist_lags, dtype=torch.long, device=device)
                hlen = torch.tensor([H], dtype=torch.long, device=device)
            else:
                hf = ho = hl = hlen = None

            expected, _ = model(cur, hf, ho, hl, hlen)
            scores = expected.squeeze(0).float().cpu().tolist()
            results.append({s: round(v, 2) for s, v in zip(_PHQ8_SYMPTOMS, scores)})
    return results


def predict_from_cache_stackattn(
    model: AidaStackAttn,
    aida_cache: dict[str, list[float]],
    run_ids_and_sessions: list[tuple[str, int]],
) -> list[dict[str, float]]:
    from EmoTrack.generative.schema import PHQ8_SYMPTOMS as _PHQ8_SYMPTOMS

    model.eval()
    device = next(model.parameters()).device
    results: list[dict[str, float]] = []

    with torch.no_grad():
        for run_id, session_index in run_ids_and_sessions:
            key = f"{run_id}::s{session_index}"
            if key not in aida_cache:
                raise KeyError(f"AIDA features not in cache: {key}")
            cur = torch.tensor([aida_cache[key]], dtype=torch.float32, device=device)

            prev = None
            has_prev = None
            is_daic = run_id.isdigit()
            if not is_daic and session_index > 1:
                pk = f"{run_id}::s{session_index - 1}"
                if pk in aida_cache:
                    prev = torch.tensor([aida_cache[pk]], dtype=torch.float32, device=device)
                    has_prev = torch.tensor([True], dtype=torch.bool, device=device)

            expected, _ = model(cur, prev, has_prev)
            scores = expected.squeeze(0).float().cpu().tolist()
            results.append({s: round(v, 2) for s, v in zip(_PHQ8_SYMPTOMS, scores)})
    return results
