import time
import torch
from sentence_transformers import SentenceTransformer
from app.core.logger import logger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingService:
    def __init__(self):
        self.model = None
        self._ready = False

    def _ensure_loaded(self):
        if self._ready:
            return
        logger.info(f"Loading embedding model: all-MiniLM-L6-v2 on {DEVICE.upper()}...")
        t0 = time.time()
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
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
        return 384


embedding_service = EmbeddingService()
