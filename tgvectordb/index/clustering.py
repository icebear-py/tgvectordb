import math

import numpy as np

from tgvectordb.utils.config import (
    CLUSTERING_THRESHOLD,
    DEFAULT_NPROBE,
    MAX_CLUSTERS,
    MIN_CLUSTERS,
)


def compute_num_clusters(total_vectors: int) -> int:
    if total_vectors < CLUSTERING_THRESHOLD:
        return 0
    n_clusters = int(math.sqrt(total_vectors))
    n_clusters = max(n_clusters, MIN_CLUSTERS)
    n_clusters = min(n_clusters, MAX_CLUSTERS)
    return n_clusters


def run_kmeans(vectors: np.ndarray, n_clusters: int) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    if len(vectors) < n_clusters:
        n_clusters = max(2, len(vectors) // 2)
    print(f"clustering {len(vectors)} vectors into {n_clusters} clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(1024, len(vectors)),
        max_iter=100,
        n_init=3,
        random_state=42,
    )
    kmeans.fit(vectors)
    print(f"clustering done. inertia: {kmeans.inertia_:.2f}")
    return kmeans.cluster_centers_.astype(np.float32)


def assign_to_nearest_cluster(vector: np.ndarray, centroids: np.ndarray) -> int:
    vector_norm = vector / (np.linalg.norm(vector) + 1e-10)
    cent_norms = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    similarities = cent_norms @ vector_norm
    return int(np.argmax(similarities))


def find_nearest_clusters(
    query_vector: np.ndarray,
    centroids: np.ndarray,
    nprobe: int = None,
) -> list:
    nprobe = nprobe or DEFAULT_NPROBE
    qvec = query_vector / (np.linalg.norm(query_vector) + 1e-10)
    cent_norms = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    similarities = cent_norms @ qvec
    if nprobe >= len(centroids):
        top_indices = np.argsort(similarities)[::-1]
    else:
        top_indices = np.argpartition(similarities, -nprobe)[-nprobe:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    result = [(int(idx), float(similarities[idx])) for idx in top_indices]
    return result
