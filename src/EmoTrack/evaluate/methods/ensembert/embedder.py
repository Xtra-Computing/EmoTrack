from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


@dataclass
class FrozenTextEmbedder:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    batch_size: int = 32
    max_length: int = 256
    pooling_strategy: str = "mean"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    device: str = ""

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("embedding batch_size must be >= 1")
        if self.max_length <= 0:
            raise ValueError("embedding max_length must be >= 1")
        if self.pooling_strategy not in {"mean", "last_token"}:
            raise ValueError("pooling_strategy must be one of: mean, last_token")
        self._tokenizer = None
        self._model = None

    @property
    def embedding_dim(self) -> int:
        self._ensure_loaded()
        return int(self._model.config.hidden_size)

    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            self._ensure_loaded()
            return torch.zeros((0, self.embedding_dim), dtype=torch.float32)
        self._ensure_loaded()
        all_embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for start in tqdm(range(0, len(texts), self.batch_size), desc="Embedding", unit="batch", leave=False):
                batch_texts = texts[start : start + self.batch_size]
                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self._model.device) for key, value in encoded.items()}
                outputs = self._model(**encoded)
                if self.pooling_strategy == "last_token":
                    pooled = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
                else:
                    pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                all_embeddings.append(F.normalize(pooled.float(), dim=1).cpu())
        return torch.cat(all_embeddings, dim=0)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        dtype = _resolve_torch_dtype(self.torch_dtype)
        model_kwargs = {"trust_remote_code": self.trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            padding_side="left" if self.pooling_strategy == "last_token" else "right",
        )
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
        self._model.eval()
        target_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(target_device)


def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = torch.sum(hidden_states * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    batch_size = hidden_states.size(0)
    if attention_mask[:, -1].sum() == batch_size:
        return hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1)
    return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths - 1]


def _resolve_torch_dtype(raw: str) -> torch.dtype | None:
    value = str(raw or "auto").strip().lower()
    if value in {"", "auto", "none"}:
        return None
    if value in {"fp32", "float32"}:
        return torch.float32
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError("torch_dtype must be one of: auto, float32, float16, bfloat16")
