"""
wraps sentence-transformers to give us a clean embed() interface.

uses intfloat/e5-small-v2 by default. if you want a different model
you can pass it in, but the dims need to be <= 384ish to fit
in telegram messages with int8 quantization.

e5 models expect a prefix on the text:
  - "query: " for search queries
  - "passage: " for documents being indexed
this matters for quality so we handle it automatically.
"""

import numpy as np
from typing import Union

from tgvectordb.utils.config import DEFAULT_MODEL_NAME, DEFAULT_DIMENSIONS


class EmbeddingModel:
    """
    lazy-loads the model on first use so import doesnt take forever.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self._model = None
        self.dimensions = None

    def _load_model(self):
        """actually load the model. only happens once."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. install it with:\n"
                "  pip install sentence-transformers"
            )

        print(f"loading embedding model: {self.model_name}")
        print("(this only happens once per session, hang tight...)")
        self._model = SentenceTransformer(self.model_name)
        self.dimensions = self._model.get_sentence_embedding_dimension()
        print(f"model loaded. dimensions: {self.dimensions}")

    def embed_query(self, text: str) -> np.ndarray:
        """
        embed a search query. adds the 'query: ' prefix for e5 models.
        returns float32 numpy array of shape (dims,)
        """
        self._load_model()

        # e5 models need this prefix for queries
        if "e5" in self.model_name.lower():
            text = f"query: {text}"

        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def embed_document(self, text: str) -> np.ndarray:
        """
        embed a document chunk. adds 'passage: ' prefix for e5.
        """
        self._load_model()

        if "e5" in self.model_name.lower():
            text = f"passage: {text}"

        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def embed_documents_batch(self, texts: list) -> np.ndarray:
        """
        embed multiple documents at once. way faster than one by one
        because it batches the GPU/CPU work.

        returns shape (n, dims)
        """
        self._load_model()

        # add prefix to all
        if "e5" in self.model_name.lower():
            texts = [f"passage: {t}" for t in texts]

        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=len(texts) > 50)
        return np.asarray(vecs, dtype=np.float32)

    def get_dimensions(self) -> int:
        """get the output dimensionality. loads model if needed."""
        self._load_model()
        return self.dimensions
