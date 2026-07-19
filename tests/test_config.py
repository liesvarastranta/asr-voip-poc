from asr_service.config import Settings


def test_default_settings():
    s = Settings()
    assert s.model_id == "small"
    assert s.device == "cuda"
    assert s.compute_type == "int8_float16"
    assert s.chunk_ms == 500
    assert s.sample_rate == 16000
    assert s.max_session_s == 300


def test_env_override(monkeypatch):
    monkeypatch.setenv("ASR_DEVICE", "cpu")
    monkeypatch.setenv("ASR_COMPUTE_TYPE", "int8")
    s = Settings()
    assert s.device == "cpu"
    assert s.compute_type == "int8"
