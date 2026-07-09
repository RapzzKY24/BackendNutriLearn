import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")


@pytest.fixture(autouse=True)
def mock_llm():
    with patch("app.services.rag_service.llm_service") as mock:
        mock.generate = AsyncMock(return_value="Ini adalah jawaban mock untuk testing.")
        mock.generate_stream = AsyncMock()
        mock.generate_stream.return_value.__aiter__.return_value = iter(["Ini ", "adalah ", "jawaban ", "stream."])
        mock.is_loaded = True
        yield mock


@pytest.fixture(autouse=True)
def mock_embedding():
    with patch("app.services.retriever_service.embedding_service") as mock:
        mock.embed.return_value = [[0.1] * 384]
        mock.embed_async = AsyncMock(return_value=[[0.1] * 384])
        mock.is_ready = True
        yield mock


@pytest.fixture(autouse=True)
def mock_retriever():
    with patch("app.services.rag_service.retriever_service") as mock:
        mock.is_ready = True
        mock.retrieve_async = AsyncMock(return_value=[
            ("Ini adalah konteks tentang gizi seimbang.", {"page": 15, "source": "test.pdf"}),
        ])
        yield mock
