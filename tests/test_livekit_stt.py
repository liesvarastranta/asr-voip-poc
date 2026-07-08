"""Tests for QwenASRSTT plugin."""
import pytest


def test_qwen_asr_stt_import():
    from livekit_agent.qwen_asr_stt import QwenASRSTT
    assert QwenASRSTT is not None


def test_qwen_asr_stt_constructor():
    from livekit_agent.qwen_asr_stt import QwenASRSTT
    stt = QwenASRSTT(endpoint="http://localhost:18001")
    assert stt.model == "Qwen/Qwen3-ASR-1.7B-hf"
    assert stt.provider == "local-gx10"
    caps = stt.capabilities
    assert caps.streaming is False
    assert caps.offline_recognize is True


def test_qwen_asr_stt_empty_buffer():
    """Empty audio buffer returns empty transcript without HTTP call."""
    import asyncio
    from livekit_agent.qwen_asr_stt import QwenASRSTT
    from livekit.agents.types import APIConnectOptions

    async def run():
        stt_inst = QwenASRSTT()
        result = await stt_inst._recognize_impl(
            [],
            language=None,
            conn_options=APIConnectOptions(),
        )
        assert result.type.value == "final_transcript"
        assert result.alternatives[0].text == ""

    asyncio.run(run())
