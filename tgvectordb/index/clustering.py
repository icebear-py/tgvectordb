"""
handles the IVF-style clustering.

when you have enough vectors (>1000), we group them into clusters
using k-means. each cluster has a centroid that represents the
"center" of that group. on search, we compare the query against
centroids to figure out which clusters to actually fetch from telegram.

this is what makes it fast - instead of downloading 50k messages,
we download maybe 500 from the 2-3 most relevant clusters.
"""

import math
import numpy as np
from typing import Optional

from tgvectordb.utils.config import (
    MIN_CLUSTERS,
    MAX_CLUSTERS,
    CLUSTERING_THRESHOLD,
    DEFAULT_NPROBE,
)


def compute_num_clusters(total_vectors: int) -> int:
    """
    figure out how many clusters to make.
    roughly sqrt(n) but with some bounds.
    """
    if total_vectors < CLUSTERING_THRESHOLD:
        return 0  # use flat search, no clustering

    n_clusters = int(math.sqrt(total_vectors))
    n_clusters = max(n_clusters, MIN_CLUSTERS)
    n_clusters = min(n_clusters, MAX_CLUSTERS)

    return n_clusters


def run_kmeans(vectors: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    cluster the vectors and return centroids.

    uses MiniBatchKMeans because its way faster than regular kmeans
    for larger datasets. the quality difference is negligible.

    args:
        vectors: shape (n, dims), float32
        n_clusters: how many clusters

    returns:
        centroids array, shape (n_clusters, dims)
    """
    from sklearn.cluster import MiniBatchKMeans

    # sanity checks
    if len(vectors) < n_clusters:
        # more clusters than vectors makes no sense, reduce
        n_clusters = max(2, len(vectors) // 2)

    print(f"clustering {len(vectors)} vectors into {n_clusters} clusters...")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(1024, len(vectors)),
        max_iter=100,
        n_init=3,  # run a few times and pick best
        random_state=42,
    )
    kmeans.fit(vectors)

    print(f"clustering done. inertia: {kmeans.inertia_:.2f}")
    return kmeans.cluster_centers_.astype(np.float32)


def assign_to_nearest_cluster(vector: np.ndarray, centroids: np.ndarray) -> int:
    """
    find which cluster a single vector belongs to.
    just picks the centroid with highest cosine similarity.
    """
    # normalize for cosine sim
    vec_norm = vector / (np.linalg.norm(vector) + 1e-10)
    cent_norms = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)

    similarities = cent_norms @ vec_norm
    return int(np.argmax(similarities))


def find_nearest_clusters(
    query_vector: np.ndarray,
    centroids: np.ndarray,
    nprobe: int = None,
) -> list:
    """
    find the top-N closest clusters to the query vector.
    these are the clusters we'll actually fetch from telegram.

    returns list of (cluster_id, similarity_score) sorted by score descending.
    """
    nprobe = nprobe or DEFAULT_NPROBE

    # normalize
    qvec = query_vector / (np.linalg.norm(query_vector) + 1e-10)
    cent_norms = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)

    similarities = cent_norms @ qvec

    # get top nprobe indices
    if nprobe >= len(centroids):
        top_indices = np.argsort(similarities)[::-1]
    else:
        # partial sort is faster than full sort when we only need top k
        top_indices = np.argpartition(similarities, -nprobe)[-nprobe:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    result = [(int(idx), float(similarities[idx])) for idx in top_indices]
    return result
