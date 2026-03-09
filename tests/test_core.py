"""
tests for the parts that dont need a telegram connection.
you can run these without any credentials.

    pytest tests/test_core.py -v
"""

import numpy as np
import json
from tgvectordb.embedding.quantizer import Quantizer
from tgvectordb.embedding.chunker import chunk_text
from tgvectordb.utils.serialization import pack_vector_message, unpack_vector_message
from tgvectordb.search.engine import cosine_similarity_batch, rank_results
from tgvectordb.index.clustering import (
    compute_num_clusters,
    assign_to_nearest_cluster,
    find_nearest_clusters,
)
from tgvectordb.index.cache import VectorCache


class TestQuantizer:
    def test_roundtrip_preserves_shape(self):
        q = Quantizer(dimensions=384)
        vec = np.random.randn(384).astype(np.float32)
        int8_vec, params = q.quantize(vec)
        restored = q.dequantize(int8_vec, params)
        assert restored.shape == (384,)

    def test_roundtrip_quality(self):
        """quantization should preserve most of the info. cosine sim > 0.95"""
        q = Quantizer(dimensions=384)
        vec = np.random.randn(384).astype(np.float32)
        int8_vec, params = q.quantize(vec)
        restored = q.dequantize(int8_vec, params)

        # cosine similarity between original and restored
        cos_sim = np.dot(vec, restored) / (np.linalg.norm(vec) * np.linalg.norm(restored))
        assert cos_sim > 0.95, f"quality too low: {cos_sim}"

    def test_int8_range(self):
        q = Quantizer(dimensions=10)
        vec = np.array([0.0, 1.0, -1.0, 0.5, -0.5, 0.3, -0.3, 0.9, -0.9, 0.1], dtype=np.float32)
        int8_vec, _ = q.quantize(vec)
        assert int8_vec.dtype == np.uint8
        assert int8_vec.min() >= 0
        assert int8_vec.max() <= 255

    def test_batch_quantize(self):
        q = Quantizer(dimensions=384)
        vecs = np.random.randn(50, 384).astype(np.float32)
        int8_vecs, params_list = q.quantize_batch(vecs)
        assert int8_vecs.shape == (50, 384)
        assert len(params_list) == 50


class TestSerialization:
    def test_pack_unpack_roundtrip(self):
        q = Quantizer(dimensions=384)
        vec = np.random.randn(384).astype(np.float32)
        int8_vec, params = q.quantize(vec)

        metadata = {"src": "test.pdf", "page": 3}
        text = "this is a test chunk about machine learning and stuff"

        msg_str = pack_vector_message(int8_vec, params, metadata, text=text)

        # should be valid json
        parsed_json = json.loads(msg_str)
        assert "v" in parsed_json
        assert "q" in parsed_json
        assert "m" in parsed_json

        # should fit in telegram message
        assert len(msg_str) <= 4096

        # unpack should give back the same data
        result = unpack_vector_message(msg_str)
        assert np.array_equal(result["vector_int8"], int8_vec)
        assert result["metadata"]["src"] == "test.pdf"
        assert result["metadata"]["text"] == text

    def test_long_text_gets_truncated(self):
        q = Quantizer(dimensions=384)
        vec = np.random.randn(384).astype(np.float32)
        int8_vec, params = q.quantize(vec)

        # make a really long text
        long_text = "word " * 2000  # way too long
        metadata = {"src": "test"}

        msg_str = pack_vector_message(int8_vec, params, metadata, text=long_text)
        assert len(msg_str) <= 4096


class TestChunker:
    def test_basic_chunking(self):
        text = "First paragraph about dogs. " * 50 + "\n\n" + "Second paragraph about cats. " * 50
        chunks = chunk_text(text, chunk_size=30, overlap=5)
        assert len(chunks) > 1
        assert all("text" in c for c in chunks)

    def test_empty_text(self):
        chunks = chunk_text("", chunk_size=100)
        assert chunks == []

    def test_source_metadata(self):
        chunks = chunk_text("Hello world this is a test.", source="test.txt")
        assert chunks[0]["src"] == "test.txt"

    def test_chunk_idx_increments(self):
        text = ("This is a paragraph. " * 30 + "\n\n") * 5
        chunks = chunk_text(text, chunk_size=20)
        indices = [c["chunk_idx"] for c in chunks]
        assert indices == sorted(indices)


class TestSearch:
    def test_cosine_similarity(self):
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vectors = np.array([
            [1.0, 0.0, 0.0],  # identical
            [0.0, 1.0, 0.0],  # orthogonal
            [-1.0, 0.0, 0.0],  # opposite
        ], dtype=np.float32)
        sims = cosine_similarity_batch(query, vectors)
        assert sims[0] > 0.99  # identical should be ~1.0
        assert abs(sims[1]) < 0.01  # orthogonal should be ~0.0
        assert sims[2] < -0.99  # opposite should be ~-1.0


class TestClustering:
    def test_num_clusters_small(self):
        assert compute_num_clusters(500) == 0  # too small for clustering

    def test_num_clusters_medium(self):
        n = compute_num_clusters(10000)
        assert 8 <= n <= 128

    def test_assign_to_nearest(self):
        centroids = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        vec = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        cluster_id = assign_to_nearest_cluster(vec, centroids)
        assert cluster_id == 0  # closest to first centroid

    def test_find_nearest_clusters(self):
        centroids = np.random.randn(16, 384).astype(np.float32)
        query = np.random.randn(384).astype(np.float32)
        result = find_nearest_clusters(query, centroids, nprobe=3)
        assert len(result) == 3
        # should be sorted by score descending
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)


class TestCache:
    def test_basic_put_get(self):
        cache = VectorCache(max_items=10)
        cache.put(1, {"test": "data"})
        assert cache.get(1) == {"test": "data"}
        assert cache.get(999) is None

    def test_eviction(self):
        cache = VectorCache(max_items=3)
        cache.put(1, {"a": 1})
        cache.put(2, {"a": 2})
        cache.put(3, {"a": 3})
        cache.put(4, {"a": 4})  # should evict 1
        assert cache.get(1) is None
        assert cache.get(2) is not None

    def test_get_many(self):
        cache = VectorCache(max_items=100)
        cache.put(1, {"x": 1})
        cache.put(3, {"x": 3})
        cached, uncached = cache.get_many([1, 2, 3, 4])
        assert set(cached.keys()) == {1, 3}
        assert set(uncached) == {2, 4}

    def test_stats(self):
        cache = VectorCache()
        cache.put(1, {})
        cache.get(1)  # hit
        cache.get(2)  # miss
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
