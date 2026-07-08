import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from asr_service.api.stream import router
from asr_service.engines.mock import MockASREngine


@pytest.fixture
def engine():
    return MockASREngine()


@pytest.fixture
def app(engine):
    app = FastAPI()
    app.state.engine = engine
    app.include_router(router, prefix="/v1/asr")
    return app


@pytest.fixture
async def client(app, engine):
    await engine.load()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_stream_basic(client):
    pcm = b"\x00\x00" * 32000
    async with client.stream(
        "POST", "/v1/asr/stream",
        content=pcm,
        headers={"Content-Type": "application/octet-stream"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = []
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
        assert "ready" in events
        assert "final" in events


@pytest.mark.asyncio
async def test_stream_headers(client):
    pcm = b"\x00\x00" * 3200
    async with client.stream(
        "POST", "/v1/asr/stream",
        content=pcm,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Language": "id",
            "X-Audio-Sample-Rate": "16000",
            "X-Audio-Channels": "1",
        },
    ) as resp:
        assert resp.status_code == 200
