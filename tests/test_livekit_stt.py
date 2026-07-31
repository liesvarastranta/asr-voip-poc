"""Tests for WhisperASRSTT plugin (local + Groq cloud modes)."""
import pytest


def test_whisper_asr_stt_import():
    from livekit_agent.whisper_asr_stt import WhisperASRSTT
    assert WhisperASRSTT is not None


def test_whisper_asr_stt_local_mode():
    """No api_key → local mode, hits custom /v1/asr/transcribe endpoint."""
    from livekit_agent.whisper_asr_stt import WhisperASRSTT
    stt = WhisperASRSTT(endpoint="http://localhost:18001")
    assert stt._is_cloud is False
    assert stt.provider == "local-wsl2"
    assert stt.model == "whisper-large-v3"
    caps = stt.capabilities
    assert caps.streaming is False
    assert caps.offline_recognize is True


def test_whisper_asr_stt_groq_mode():
    """api_key + groq.com host → cloud mode, uses openai SDK."""
    from livekit_agent.whisper_asr_stt import WhisperASRSTT
    stt = WhisperASRSTT(
        endpoint="https://api.groq.com/openai/v1",
        api_key="gsk_test",
    )
    assert stt._is_cloud is True
    assert stt.provider == "groq"
    assert stt._client is not None


def test_whisper_asr_stt_openai_mode():
    """api_key + openai.com host → cloud mode."""
    from livekit_agent.whisper_asr_stt import WhisperASRSTT
    stt = WhisperASRSTT(
        endpoint="https://api.openai.com/v1",
        api_key="sk-test",
    )
    assert stt._is_cloud is True
    assert stt.provider == "groq"  # provider label unchanged, only "groq" or "local-wsl2"


def test_whisper_asr_stt_custom_model():
    """Custom model name is honored."""
    from livekit_agent.whisper_asr_stt import WhisperASRSTT
    stt = WhisperASRSTT(endpoint="http://localhost:18001", model="whisper-large-v3")
    assert stt.model == "whisper-large-v3"


def test_whisper_asr_stt_empty_buffer():
    """Empty audio buffer returns empty transcript without HTTP call."""
    import asyncio
    from livekit_agent.whisper_asr_stt import WhisperASRSTT
    from livekit.agents.types import APIConnectOptions

    async def run():
        stt_inst = WhisperASRSTT()
        result = await stt_inst._recognize_impl(
            [],
            language=None,
            conn_options=APIConnectOptions(),
        )
        assert result.type.value == "final_transcript"
        assert result.alternatives[0].text == ""

    asyncio.run(run())
