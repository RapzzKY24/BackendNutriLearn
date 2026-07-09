import time
import torch
from sentence_transformers import SentenceTransformer
from app.core.logger import logger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384


class EmbeddingService:
    def __init__(self):
        self.model = None
        self._ready = False

    def _ensure_loaded(self):
        if self._ready:
            return
        logger.info(f"Loading embedding model: {MODEL_NAME} on {DEVICE.upper()}...")
        t0 = time.time()
        self.model = SentenceTransformer(
            MODEL_NAME,
            device=DEVICE,
        )
        elapsed = time.time() - t0
        logger.info(f"Embedding model loaded in {elapsed:.2f}s")
        self._ready = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        return self.model.encode(texts, show_progress_bar=False).tolist()

    @property
    def dimension(self) -> int:
        return EMBED_DIM


embedding_service = EmbeddingService()
