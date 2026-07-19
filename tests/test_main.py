import pytest
from httpx import AsyncClient, ASGITransport
from asr_service.main import create_app
from asr_service.engines.mock import MockASREngine


@pytest.fixture
async def client():
    engine = MockASREngine()
    await engine.load()
    app = create_app(engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health_via_main(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


@pytest.mark.asyncio
async def test_openapi_docs(client):
    r = await client.get("/docs")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_openapi_json(client):
    r = await client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    paths = schema["paths"]
    assert "/v1/asr/stream" in paths
    assert "/v1/asr/transcribe" in paths
    assert "/v1/models" in paths
    assert "/health" in paths
