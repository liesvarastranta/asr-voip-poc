"""Tests for WhisperASRSTT plugin."""
import pytest


def test_whisper_asr_stt_import():
    from livekit_agent.whisper_asr_stt import WhisperASRSTT
    assert WhisperASRSTT is not None


def test_whisper_asr_stt_constructor():
    from livekit_agent.whisper_asr_stt import WhisperASRSTT
    stt = WhisperASRSTT(endpoint="http://localhost:18001")
    assert stt.model == "openai/whisper-small"
    assert stt.provider == "local-wsl2"
    caps = stt.capabilities
    assert caps.streaming is False
    assert caps.offline_recognize is True


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
