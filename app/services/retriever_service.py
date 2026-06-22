import os
import chromadb
from chromadb.config import Settings

os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
from app.core.config import settings
from app.core.logger import logger
from app.services.embedding_service import embedding_service

COLLECTION_NAME = "permenkes_gizi"


class RetrieverService:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = None
        self._ready = False
        self._try_load()

    def _try_load(self):
        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
            count = self.collection.count()
            self._ready = count > 0
            if self._ready:
                logger.info(f"Retriever ready — ChromaDB collection '{COLLECTION_NAME}' has {count} documents")
            else:
                logger.warning(f"Collection '{COLLECTION_NAME}' is empty")
        except Exception:
            logger.warning("Retriever not ready — ChromaDB belum di-populate. Jalankan ingest.py dulu.")

    def retrieve(self, question: str, k: int | None = None) -> list[tuple[str, dict]]:
        if not self._ready:
            return []
        k = k or settings.top_k
        query_embedding = embedding_service.embed([question])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        return list(zip(docs, metas))

    @property
    def is_ready(self) -> bool:
        return self._ready


retriever_service = RetrieverService()
