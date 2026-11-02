from __future__ import annotations

import torch
import torch.nn.functional as F


def top_k_indices(
    document_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    *,
    k: int,
) -> list[int]:
    if k <= 0:
        raise ValueError("k must be >= 1")
    if document_embeddings.ndim != 2:
        raise ValueError("document_embeddings must have shape (N, D)")
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must have shape (D,)")
    if document_embeddings.size(0) == 0:
        return []
    docs = F.normalize(document_embeddings.float(), dim=1)
    query = F.normalize(query_embedding.float(), dim=0)
    scores = torch.matmul(docs, query)
    top_count = min(k, document_embeddings.size(0))
    return torch.topk(scores, k=top_count).indices.cpu().tolist()


def gather_top_k_embeddings(
    document_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    *,
    k: int,
    retrieval_document_embeddings: torch.Tensor | None = None,
) -> torch.Tensor:
    if k <= 0:
        raise ValueError("k must be >= 1")
    if document_embeddings.ndim != 2:
        raise ValueError("document_embeddings must have shape (N, D)")
    retrieval_embeddings = retrieval_document_embeddings if retrieval_document_embeddings is not None else document_embeddings
    if retrieval_embeddings.ndim != 2:
        raise ValueError("retrieval_document_embeddings must have shape (N, D)")
    if retrieval_embeddings.size(0) != document_embeddings.size(0):
        raise ValueError("retrieval_document_embeddings must have the same row count as document_embeddings")

    embedding_dim = int(document_embeddings.size(1))
    output = torch.zeros(k, embedding_dim, dtype=torch.float32)
    indices = top_k_indices(retrieval_embeddings, query_embedding, k=k)
    for output_index, document_index in enumerate(indices):
        output[output_index] = document_embeddings[document_index].float()
    return output
