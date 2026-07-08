from asr_service.config import Settings

def test_default_settings():
    s = Settings()
    assert s.model_id == "Qwen/Qwen3-ASR-1.7B-hf"
    assert s.device == "cuda:0"
    assert s.dtype == "bfloat16"
    assert s.chunk_ms == 500
    assert s.sample_rate == 16000
    assert s.max_session_s == 300

def test_env_override(monkeypatch):
    monkeypatch.setenv("ASR_DEVICE", "cpu")
    monkeypatch.setenv("ASR_DTYPE", "float32")
    s = Settings()
    assert s.device == "cpu"
    assert s.dtype == "float32"
