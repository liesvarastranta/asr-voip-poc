import pytest


def test_can_import_faster_whisper_engine():
    from asr_service.engines.faster_whisper import FasterWhisperEngine
    assert FasterWhisperEngine is not None


def test_faster_whisper_constructor():
    from asr_service.engines.faster_whisper import FasterWhisperEngine
    engine = FasterWhisperEngine(
        model_id="large-v3", device="cuda", compute_type="int8_float16"
    )
    assert engine.model_id == "large-v3"
    assert engine.device == "cuda"
    assert engine.compute_type == "int8_float16"
    assert engine.is_loaded is False


@pytest.mark.skip(
    reason="requires CUDA GPU + faster-whisper installed — run on WSL2 with GPU"
)
def test_faster_whisper_load_and_transcribe():
    import asyncio
    from asr_service.engines.faster_whisper import FasterWhisperEngine

    async def run():
        engine = FasterWhisperEngine(
            model_id="large-v3", device="cuda", compute_type="int8_float16"
        )
        await engine.load()
        assert engine.is_loaded
        # Requires a test.wav fixture in repo root
        result = await engine.transcribe_file("test.wav", language="id")
        assert "text" in result
        assert "language" in result
        assert "duration_ms" in result
        assert "processing_ms" in result

    asyncio.run(run())
