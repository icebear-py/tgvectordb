import numpy as np

from tgvectordb.embedding.quantizer import Quantizer


def cosine_similarity_batch(query_vec: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    v_normed = vectors / norms
    sims = v_normed @ q_norm
    return sims


def rank_results(
    query_vec: np.ndarray,
    candidates: dict,
    quantizer: Quantizer,
    top_k: int = 5,
    filter_fn=None,
) -> list:
    if not candidates:
        return []
    message_ids = []
    vectors = []
    metas = []
    for message_id, data in candidates.items():
        meta = data["metadata"]
        if filter_fn and not filter_fn(meta):
            continue
        float_vec = quantizer.dequantize(data["vector_int8"], data["quant_params"])
        message_ids.append(message_id)
        vectors.append(float_vec)
        metas.append(meta)
    if not vectors:
        return []
    vectors_array = np.array(vectors, dtype=np.float32)
    similarities = cosine_similarity_batch(query_vec, vectors_array)
    if top_k >= len(similarities):
        sorted_indices = np.argsort(similarities)[::-1]
    else:
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        sorted_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    results = []
    for idx in sorted_indices[:top_k]:
        score = float(similarities[idx])
        if score < 0.01:
            break
        result = {
            "text": metas[idx].get("text", ""),
            "score": round(score, 4),
            "metadata": {k: v for k, v in metas[idx].items() if k != "text"},
            "message_id": message_ids[idx],
        }
        results.append(result)
    return results
