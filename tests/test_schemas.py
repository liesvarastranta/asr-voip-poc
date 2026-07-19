from asr_service.api.schemas import TranscribeResponse, HealthResponse, SSEPartial, SSEFinal, SSEError

def test_transcribe_response():
    r = TranscribeResponse(text="Halo", language="id", duration_ms=1000, processing_ms=500)
    assert r.text == "Halo"
    assert r.language == "id"
    assert r.processing_ms == 500

def test_health_response():
    h = HealthResponse(status="ok", model_loaded=True, model="Qwen/Qwen3-ASR-1.7B-hf", device="cuda:0")
    assert h.status == "ok"
    assert h.model_loaded is True

def test_sse_partial():
    p = SSEPartial(text="Halo", is_final=False, t_ms=500)
    assert p.is_final is False
    data = p.model_dump_json()
    assert "Halo" in data

def test_sse_final():
    f = SSEFinal(text="Halo.", is_final=True, t_ms=1000, language="id")
    assert f.is_final is True

def test_sse_error():
    e = SSEError(code="AUDIO_FORMAT", message="bad format")
    assert e.code == "AUDIO_FORMAT"

def test_transcribe_response_serialization():
    r = TranscribeResponse(text="Halo", language="id", duration_ms=1000)
    d = r.model_dump()
    assert d["processing_ms"] is None

def test_health_serialization():
    h = HealthResponse(status="ok", model_loaded=True)
    d = h.model_dump()
    assert d["model"] is None
    assert d["device"] is None
