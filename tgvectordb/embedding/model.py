import numpy as np

from tgvectordb.utils.config import DEFAULT_MODEL_NAME


class EmbeddingModel:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self._model = None
        self.dimensions = None

    def _load_model(self):
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
        self._load_model()
        if "e5" in self.model_name.lower():
            text = f"query: {text}"
        vector = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32)

    def embed_document(self, text: str) -> np.ndarray:
        self._load_model()
        if "e5" in self.model_name.lower():
            text = f"passage: {text}"
        vector = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32)

    def embed_documents_batch(self, texts: list) -> np.ndarray:
        self._load_model()
        if "e5" in self.model_name.lower():
            texts = [f"passage: {t}" for t in texts]
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=len(texts) > 50
        )
        return np.asarray(vectors, dtype=np.float32)

    def get_dimensions(self) -> int:
        self._load_model()
        return self.dimensions
