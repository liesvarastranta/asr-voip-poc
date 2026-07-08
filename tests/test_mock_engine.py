import pytest
from asr_service.engines.mock import MockASREngine

@pytest.mark.asyncio
async def test_mock_initial_state():
    engine = MockASREngine()
    assert engine.is_loaded is False

@pytest.mark.asyncio
async def test_mock_load():
    engine = MockASREngine()
    await engine.load()
    assert engine.is_loaded is True

@pytest.mark.asyncio
async def test_mock_transcribe_file():
    engine = MockASREngine()
    await engine.load()
    result = await engine.transcribe_file("fake.wav", language="id")
    assert "text" in result
    assert result["text"] == "mock transcription"
    assert result["language"] == "id"
    assert "duration_ms" in result
    assert "processing_ms" in result

@pytest.mark.asyncio
async def test_mock_infer_chunk_partial():
    engine = MockASREngine()
    await engine.load()
    pcm = b"\x00\x00" * 16000
    text = await engine.infer_chunk(pcm, is_final=False)
    assert text == "mock partial"

@pytest.mark.asyncio
async def test_mock_infer_chunk_final():
    engine = MockASREngine()
    await engine.load()
    text = await engine.infer_chunk(b"", is_final=True)
    assert text == "mock final"
