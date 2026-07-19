import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from asr_service.engines.mock import MockASREngine
from asr_service.api import health, transcribe, stream


@pytest.fixture
def engine():
    return MockASREngine()


@pytest.fixture
def app(engine):
    app = FastAPI()
    app.state.engine = engine
    app.include_router(health.router)
    app.include_router(transcribe.router, prefix="/v1/asr")
    app.include_router(stream.router, prefix="/v1/asr")
    return app


@pytest.fixture
async def client(app, engine):
    await engine.load()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
