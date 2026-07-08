from asr_service.api.sse import format_sse_event

def test_partial_event():
    result = format_sse_event("partial", {"text": "Halo", "is_final": False, "t_ms": 500})
    assert "event: partial" in result
    assert '"text": "Halo"' in result
    assert '"is_final": false' in result
    assert result.endswith("\n\n")

def test_final_event():
    result = format_sse_event("final", {"text": "Halo.", "is_final": True, "t_ms": 1000, "language": "id"})
    assert "event: final" in result
    assert '"is_final": true' in result

def test_error_event():
    result = format_sse_event("error", {"code": "AUDIO_FORMAT", "message": "bad format"})
    assert "event: error" in result
    assert '"code": "AUDIO_FORMAT"' in result

def test_ready_event():
    result = format_sse_event("ready", {})
    assert "event: ready" in result

def test_newline_terminated():
    result = format_sse_event("test", {"key": "value"})
    assert result.endswith("\n\n")
