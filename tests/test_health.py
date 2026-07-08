import pytest
from httpx import AsyncClient, ASGITransport
from asr_service.engines.mock import MockASREngine


@pytest.mark.asyncio
async def test_health_ok(client):
    r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


@pytest.mark.asyncio
async def test_health_not_loaded(app):
    engine2 = MockASREngine()
    app.state.engine = engine2
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["model_loaded"] is False


@pytest.mark.asyncio
async def test_models(client):
    r = await client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert len(data["models"]) == 1
    assert data["models"][0]["status"] == "loaded"
