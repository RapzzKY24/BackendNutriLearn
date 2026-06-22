import os
import time
import chromadb
from chromadb.config import Settings

os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
from pypdf import PdfReader
from app.core.config import settings
from app.core.logger import logger
from app.services.embedding_service import embedding_service

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "permenkes_gizi"


def _chunk_page(text: str, page_num: int) -> list[tuple[str, int]]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            last_period = text.rfind(".", start, end)
            if last_period > start + CHUNK_SIZE // 2:
                end = last_period + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, page_num))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def run_ingestion():
    t0 = time.time()
    logger.info(f"Loading PDF: {settings.pdf_path}")

    reader = PdfReader(settings.pdf_path)
    total_pages = len(reader.pages)
    logger.info(f"PDF loaded: {total_pages} pages")

    all_chunks = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text.strip():
            chunks = _chunk_page(text, i)
            all_chunks.extend(chunks)

    logger.info(f"Total chunks: {len(all_chunks)}")

    texts = [c[0] for c in all_chunks]
    page_nums = [c[1] for c in all_chunks]

    logger.info("Generating embeddings...")
    embed_t0 = time.time()
    embeddings = embedding_service.embed(texts)
    logger.info(f"Embeddings generated in {time.time() - embed_t0:.2f}s")

    logger.info("Storing in ChromaDB...")
    client = chromadb.PersistentClient(
        path=settings.chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [str(i) for i in range(len(texts))]
    metadatas = [
        {"page": pn, "source": "Permenkes No 41 Tahun 2014"}
        for pn in page_nums
    ]

    batch_size = 166
    for i in range(0, len(texts), batch_size):
        end = min(i + batch_size, len(texts))
        collection.add(
            ids=ids[i:end],
            documents=texts[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )
        logger.info(f"  Stored {end}/{len(texts)} chunks")

    elapsed = time.time() - t0
    logger.info(f"Ingestion complete in {elapsed:.2f}s")
    logger.info(f"Collection '{COLLECTION_NAME}' has {collection.count()} documents")
