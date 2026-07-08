import struct
import pytest
from httpx import AsyncClient, ASGITransport
from asr_service.engines.mock import MockASREngine


def make_minimal_wav(sample_rate=16000, duration_sec=0.5):
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
async def test_transcribe_wav(client):
    wav = make_minimal_wav()
    r = await client.post(
        "/v1/asr/transcribe",
        files={"file": ("test.wav", wav, "audio/wav")},
        data={"language": "id"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "text" in data
    assert data["text"] == "mock transcription"
    assert data["language"] == "id"


@pytest.mark.asyncio
async def test_transcribe_no_file(client):
    r = await client.post("/v1/asr/transcribe")
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_transcribe_engine_not_loaded(app):
    engine2 = MockASREngine()
    app.state.engine = engine2
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/v1/asr/transcribe",
            files={"file": ("test.wav", make_minimal_wav(), "audio/wav")},
        )
        assert r.status_code == 503
