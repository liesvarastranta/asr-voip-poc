"""Full API integration test — uses mock engine, verifies contract."""
import json
import struct
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from asr_service.main import create_app
from asr_service.engines.mock import MockASREngine


@pytest.fixture
async def client():
    engine = MockASREngine()
    app = create_app(engine)
    await engine.load()  # ponytail: ASGITransport bypasses lifespan, match conftest.py pattern
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def make_wav(sample_rate=16000, duration_sec=1.0):
    num_samples = int(sample_rate * duration_sec)
    data_size = num_samples * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", data_size,
    )
    return header + b"\x00\x00" * num_samples


@pytest.mark.asyncio
async def test_full_flow_health_transcribe(client):
    # Health
    r = await client.get("/health")
    assert r.status_code == 200
    health = r.json()
    assert health["status"] == "ok"
    assert health["model_loaded"] is True

    # Models
    r = await client.get("/v1/models")
    assert r.status_code == 200
    models = r.json()
    assert len(models["models"]) == 1
    assert models["models"][0]["status"] == "loaded"

    # Transcribe
    r = await client.post(
        "/v1/asr/transcribe",
        files={"file": ("test.wav", make_wav(), "audio/wav")},
        data={"language": "id"},
    )
    assert r.status_code == 200
    transcribe = r.json()
    assert transcribe["text"] == "mock transcription"
    assert transcribe["language"] == "id"
    assert transcribe["duration_ms"] > 0


@pytest.mark.asyncio
async def test_stream_events_format(client):
    pcm = b"\x00\x00" * 32000  # 1 second silence
    async with client.stream(
        "POST", "/v1/asr/stream",
        content=pcm,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Audio-Sample-Rate": "16000",
            "X-Audio-Channels": "1",
            "X-Audio-Format": "s16le",
            "X-Language": "id",
        },
    ) as resp:
        assert resp.status_code == 200
        events = []
        data_by_event = {}
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            elif line.startswith("data:"):
                data_by_event.setdefault(events[-1], []).append(
                    json.loads(line.split(":", 1)[1].strip())
                )

        assert "ready" in events
        assert "final" in events
        final = data_by_event["final"][0]
        assert final["is_final"] is True
        assert final["language"] == "id"
        assert "text" in final


@pytest.mark.asyncio
async def test_stream_audio_format_rejection(client):
    pcm = b"\x00\x00" * 3200
    async with client.stream(
        "POST", "/v1/asr/stream",
        content=pcm,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Audio-Sample-Rate": "44100",
            "X-Audio-Channels": "2",
            "X-Audio-Format": "float32",
        },
    ) as resp:
        events = []
        data_by_event = {}
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            elif line.startswith("data:"):
                data_by_event.setdefault(events[-1], []).append(
                    json.loads(line.split(":", 1)[1].strip())
                )
        assert "error" in events
        error_data = data_by_event["error"][0]
        assert error_data["code"] == "AUDIO_FORMAT"


@pytest.mark.asyncio
async def test_openapi_schema(client):
    r = await client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    paths = schema["paths"]
    assert "/v1/asr/stream" in paths
    assert "/v1/asr/transcribe" in paths
    assert "/v1/models" in paths
    assert "/health" in paths


@pytest.mark.asyncio
async def test_transcribe_file_size_limit(client):
    huge_data = b"\x00" * (100 * 1024 * 1024 + 1)  # 100MB + 1
    r = await client.post(
        "/v1/asr/transcribe",
        files={"file": ("huge.wav", huge_data, "audio/wav")},
        data={"language": "id"},
    )
    assert r.status_code == 413


@pytest.mark.asyncio
async def test_transcribe_no_file_422(client):
    r = await client.post("/v1/asr/transcribe")
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_404_unknown_route(client):
    r = await client.get("/nonexistent")
    assert r.status_code == 404
