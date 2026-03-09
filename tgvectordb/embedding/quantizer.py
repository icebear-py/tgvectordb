"""
quantizer - shrinks float32 vectors down to int8 so they fit in telegram messages.

the math is straightforward:
    1. find the min and max of the vector
    2. scale everything to 0-255 range
    3. store as uint8 + remember the min/scale for reconstruction

quality loss is about 1-3% on cosine similarity. thats basically nothing
for RAG use cases where you're grabbing top 5-10 results anyway.
"""

import numpy as np


class Quantizer:
    """handles the float32 <-> int8 conversion for vectors."""

    def __init__(self, dimensions: int = 384):
        self.dims = dimensions

    def quantize(self, vector: np.ndarray) -> tuple:
        """
        compress a float32 vector to uint8.

        args:
            vector: numpy float32 array, shape (dims,)

        returns:
            (int8_vector, quant_params)
            where quant_params = (min_val, scale) needed to reconstruct
        """
        vec = np.asarray(vector, dtype=np.float32).flatten()

        if len(vec) != self.dims:
            raise ValueError(
                f"expected {self.dims} dims but got {len(vec)}. "
                f"did you change the embedding model?"
            )

        min_val = float(vec.min())
        max_val = float(vec.max())

        # edge case - if all values are the same (weird but possible)
        if max_val - min_val < 1e-10:
            return np.zeros(self.dims, dtype=np.uint8), (min_val, 1.0)

        scale = (max_val - min_val) / 255.0
        quantized = np.round((vec - min_val) / scale).astype(np.uint8)

        return quantized, (min_val, scale)

    def dequantize(self, int8_vec: np.ndarray, quant_params: tuple) -> np.ndarray:
        """
        reconstruct float32 vector from uint8 + params.
        wont be exactly the same as original but close enough.
        """
        min_val, scale = quant_params
        return (int8_vec.astype(np.float32) * scale) + min_val

    def quantize_batch(self, vectors: np.ndarray) -> tuple:
        """
        quantize a whole batch at once. each vector gets its own min/scale
        because different vectors have different ranges.

        args:
            vectors: shape (n, dims)

        returns:
            (int8_vectors, list_of_quant_params)
        """
        n = vectors.shape[0]
        quantized_vecs = np.zeros((n, self.dims), dtype=np.uint8)
        params_list = []

        for i in range(n):
            qvec, params = self.quantize(vectors[i])
            quantized_vecs[i] = qvec
            params_list.append(params)

        return quantized_vecs, params_list
