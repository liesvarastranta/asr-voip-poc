import pytest

def test_can_import_qwen_engine():
    from asr_service.engines.qwen_asr import Qwen3ASREngine
    assert Qwen3ASREngine is not None

@pytest.mark.skip(reason="requires GPU and transformers from source — run on GX10")
def test_qwen_load_and_infer():
    import asyncio
    from asr_service.engines.qwen_asr import Qwen3ASREngine
    async def run():
        engine = Qwen3ASREngine(model_id="Qwen/Qwen3-ASR-1.7B-hf", device="cuda:0", dtype="bfloat16")
        await engine.load()
        assert engine.is_loaded
        result = await engine.transcribe_file("test.wav", language="id")
        assert "text" in result
    asyncio.run(run())

def test_qwen_constructor():
    from asr_service.engines.qwen_asr import Qwen3ASREngine
    engine = Qwen3ASREngine(model_id="test/model", device="cpu", dtype="float32")
    assert engine.model_id == "test/model"
    assert engine.device == "cpu"
    assert engine.dtype_str == "float32"
    assert engine.is_loaded is False
