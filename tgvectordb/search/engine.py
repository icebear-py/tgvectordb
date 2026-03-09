"""
the actual search logic.

query flow:
  1. embed the query text
  2. find nearest clusters via centroids (local, fast)
  3. fetch vectors from those clusters (telegram, the slow part)
  4. compute cosine similarity against all fetched vectors
  5. return top-k results

if we dont have enough vectors for clustering yet,
falls back to flat search (just fetch everything and compare).
"""

import numpy as np
from typing import Optional

from tgvectordb.embedding.quantizer import Quantizer
from tgvectordb.index.clustering import find_nearest_clusters
from tgvectordb.utils.config import DEFAULT_NPROBE, CLUSTERING_THRESHOLD


def cosine_similarity_batch(query_vec: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    compute cosine similarity between a query and a batch of vectors.
    query_vec: shape (dims,)
    vectors: shape (n, dims)
    returns: shape (n,) array of similarities
    """
    # normalize query
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)

    # normalize all vectors at once
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # avoid division by zero
    v_normed = vectors / norms

    # dot product = cosine sim (since both are normalized)
    sims = v_normed @ q_norm
    return sims


def rank_results(
    query_vec: np.ndarray,
    candidates: dict,
    quantizer: Quantizer,
    top_k: int = 5,
    filter_fn=None,
) -> list:
    """
    take a bunch of candidate vectors, score them against the query,
    and return the top k.

    candidates: {msg_id: {"vector_int8": ..., "quant_params": ..., "metadata": ...}}
    filter_fn: optional function that takes metadata dict and returns True to keep

    returns list of dicts sorted by score:
    [{"text": "...", "score": 0.94, "metadata": {...}, "message_id": 123}, ...]
    """
    if not candidates:
        return []

    msg_ids = []
    vectors = []
    metas = []

    for mid, data in candidates.items():
        meta = data["metadata"]

        # apply filter if provided
        if filter_fn and not filter_fn(meta):
            continue

        # dequantize back to float32 for accurate comparison
        float_vec = quantizer.dequantize(data["vector_int8"], data["quant_params"])
        msg_ids.append(mid)
        vectors.append(float_vec)
        metas.append(meta)

    if not vectors:
        return []

    vectors_array = np.array(vectors, dtype=np.float32)
    similarities = cosine_similarity_batch(query_vec, vectors_array)

    # get top k indices
    if top_k >= len(similarities):
        sorted_indices = np.argsort(similarities)[::-1]
    else:
        # faster partial sort
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        sorted_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    results = []
    for idx in sorted_indices[:top_k]:
        score = float(similarities[idx])
        if score < 0.01:
            # basically no similarity, dont include junk
            break

        result = {
            "text": metas[idx].get("text", ""),
            "score": round(score, 4),
            "metadata": {k: v for k, v in metas[idx].items() if k != "text"},
            "message_id": msg_ids[idx],
        }
        results.append(result)

    return results
