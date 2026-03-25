import numpy as np


class Quantizer:
    def __init__(self, dimensions: int = 384):
        self.dims = dimensions

    def quantize(self, vector: np.ndarray) -> tuple:
        vector = np.asarray(vector, dtype=np.float32).flatten()
        if len(vector) != self.dims:
            raise ValueError(
                f"expected {self.dims} dims but got {len(vector)}. "
                f"did you change the embedding model?"
            )
        min_val = float(vector.min())
        max_val = float(vector.max())
        if max_val - min_val < 1e-10:
            return np.zeros(self.dims, dtype=np.uint8), (min_val, 1.0)
        scale = (max_val - min_val) / 255.0
        quantized = np.round((vector - min_val) / scale).astype(np.uint8)
        return quantized, (min_val, scale)

    def dequantize(self, int8_vector: np.ndarray, quant_params: tuple) -> np.ndarray:
        min_val, scale = quant_params
        return (int8_vector.astype(np.float32) * scale) + min_val

    def quantize_batch(self, vectors: np.ndarray) -> tuple:
        n = vectors.shape[0]
        quantized_vecs = np.zeros((n, self.dims), dtype=np.uint8)
        params_list = []
        for i in range(n):
            qvec, params = self.quantize(vectors[i])
            quantized_vecs[i] = qvec
            params_list.append(params)
        return quantized_vecs, params_list
