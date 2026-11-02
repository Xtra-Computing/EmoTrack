from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EnsemBERTModelConfig:
    embedding_dim: int
    max_posts: int = 20
    dense_units: int = 128
    num_heads: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    out_loss_weight: float = 1.0
    soft_loss_weight: float = 1.0
    meta_loss_weight: float = 1.0


@dataclass(frozen=True)
class EnsemBERTTrainingResult:
    best_epoch: int
    best_val_loss: float
    epochs_ran: int
    train_size: int
    validation_size: int
    history: list[dict[str, float]]


class ScaledDotAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(k.size(-1)))
        weights = torch.softmax(logits, dim=-1)
        return torch.matmul(weights, v)


class KerasStyleMultiHeadAttention(nn.Module):
    """Keras MultiHeadAttention equivalent for key_dim=embedding_dim."""

    def __init__(self, *, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be >= 1")
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.projection_dim = embedding_dim * num_heads
        self.q_proj = nn.Linear(embedding_dim, self.projection_dim)
        self.k_proj = nn.Linear(embedding_dim, self.projection_dim)
        self.v_proj = nn.Linear(embedding_dim, self.projection_dim)
        self.out_proj = nn.Linear(self.projection_dim, embedding_dim)
        self.attention = ScaledDotAttention()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size = query.size(0)
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.embedding_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.embedding_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.embedding_dim).transpose(1, 2)
        attended = self.attention(q, k, v).transpose(1, 2).contiguous()
        return self.out_proj(attended.view(batch_size, -1, self.projection_dim))


class EnsemBERTModel(nn.Module):
    def __init__(self, config: EnsemBERTModelConfig) -> None:
        super().__init__()
        self.config = config
        self.self_attention = KerasStyleMultiHeadAttention(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
        )
        self.cross_attention = KerasStyleMultiHeadAttention(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
        )
        self.dense = nn.Linear(config.embedding_dim * 3, config.dense_units)
        self.out_head = nn.Linear(config.dense_units, 4)
        self.meta_head = nn.Linear(8, 4)

    def forward(
        self,
        reddit_embeddings: torch.Tensor,
        description_embedding: torch.Tensor,
        choice_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        attended_posts = self.self_attention(reddit_embeddings, reddit_embeddings, reddit_embeddings)
        pooled_posts = attended_posts.mean(dim=1)
        description_query = description_embedding.unsqueeze(1)
        weighted_choice = self.cross_attention(description_query, choice_embeddings, choice_embeddings).squeeze(1)
        merged = torch.cat([pooled_posts, description_embedding, weighted_choice], dim=1)
        dense = F.relu(self.dense(merged))
        out_logits = self.out_head(dense)

        cosine_similarities = F.cosine_similarity(
            pooled_posts.unsqueeze(1),
            choice_embeddings,
            dim=-1,
        )
        soft_probs = torch.softmax(cosine_similarities, dim=-1)
        meta_logits = self.meta_head(torch.cat([out_logits, soft_probs], dim=1))
        return {
            "out_logits": out_logits,
            "soft_probs": soft_probs,
            "meta_logits": meta_logits,
        }


def ensembert_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    config: EnsemBERTModelConfig,
) -> torch.Tensor:
    out_loss = F.cross_entropy(outputs["out_logits"], labels)
    soft_loss = F.nll_loss(torch.log(outputs["soft_probs"].clamp(min=1e-7)), labels)
    meta_loss = F.cross_entropy(outputs["meta_logits"], labels)
    return config.out_loss_weight * out_loss + config.soft_loss_weight * soft_loss + config.meta_loss_weight * meta_loss


def train_ensembert_model(
    model: EnsemBERTModel,
    *,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: EnsemBERTModelConfig,
    num_epochs: int,
    patience: int,
    device: torch.device,
) -> EnsemBERTTrainingResult:
    if num_epochs <= 0:
        raise ValueError("num_epochs must be >= 1")
    if patience <= 0:
        raise ValueError("patience must be >= 1")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        train_loss = _run_epoch(model, loader=train_loader, config=config, device=device, optimizer=optimizer)
        with torch.no_grad():
            validation_loss = _run_epoch(model, loader=validation_loader, config=config, device=device, optimizer=None)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return EnsemBERTTrainingResult(
        best_epoch=best_epoch,
        best_val_loss=float(best_val_loss),
        epochs_ran=len(history),
        train_size=len(train_loader.dataset),
        validation_size=len(validation_loader.dataset),
        history=history,
    )


def predict_item_classes(
    model: EnsemBERTModel,
    *,
    loader: DataLoader,
    device: torch.device,
    head: str,
) -> list[int]:
    if head not in {"out", "soft", "meta"}:
        raise ValueError("prediction head must be one of: out, soft, meta")
    model.to(device)
    model.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["reddit_embeddings"].to(device),
                batch["description_embedding"].to(device),
                batch["choice_embeddings"].to(device),
            )
            if head == "out":
                values = outputs["out_logits"].argmax(dim=-1)
            elif head == "soft":
                values = outputs["soft_probs"].argmax(dim=-1)
            else:
                values = outputs["meta_logits"].argmax(dim=-1)
            predictions.extend(int(value) for value in values.cpu().tolist())
    return predictions


def _run_epoch(
    model: EnsemBERTModel,
    *,
    loader: DataLoader,
    config: EnsemBERTModelConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    total_loss = 0.0
    total_count = 0
    if optimizer is None:
        model.eval()
    else:
        model.train()
    for batch in loader:
        labels = batch["label"].to(device)
        outputs = model(
            batch["reddit_embeddings"].to(device),
            batch["description_embedding"].to(device),
            batch["choice_embeddings"].to(device),
        )
        loss = ensembert_loss(outputs, labels, config)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_count = int(labels.numel())
        total_loss += float(loss.item()) * batch_count
        total_count += batch_count
    if total_count == 0:
        raise ValueError("Cannot run EnsemBERT epoch on an empty loader")
    return total_loss / total_count
