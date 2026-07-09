import pytest
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture(autouse=True)
def mock_app_startup():
    with patch("app.main.llm_service.ensure_loaded", AsyncMock()):
        yield


@pytest.fixture
def client():
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_root(client):
    resp = await client.get("/")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_chat(client):
    resp = await client.post(
        "/chat",
        json={"question": "Apa itu gizi seimbang?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data


@pytest.mark.asyncio
async def test_chat_stream(client):
    resp = await client.post(
        "/chat/stream",
        json={"question": "Apa itu gizi seimbang?"},
    )
    assert resp.status_code == 200
    content = resp.text
    assert len(content) > 0


@pytest.mark.asyncio
async def test_bmi(client):
    resp = await client.post(
        "/bmi",
        json={"weight": 65, "height": 170},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["category"] == "Normal"


@pytest.mark.asyncio
async def test_bmi_invalid(client):
    resp = await client.post(
        "/bmi",
        json={"weight": -1, "height": 0},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_invalid_input(client):
    resp = await client.post(
        "/chat",
        json={"question": ""},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_cache_flush(client):
    resp = await client.post("/admin/cache/flush")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True


@pytest.mark.asyncio
async def test_chat_with_session(client):
    resp = await client.post(
        "/chat",
        json={"question": "Apa itu gizi?", "session_id": "test-123"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_api_key_auth(client):
    with patch("app.main.settings.api_key", "secret-123"):
        resp = await client.post(
            "/chat",
            json={"question": "test"},
        )
        assert resp.status_code == 401

        resp = await client.post(
            "/chat",
            json={"question": "test"},
            headers={"X-API-Key": "secret-123"},
        )
        assert resp.status_code == 200
