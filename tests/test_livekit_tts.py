"""Tests for F5TTSPlugin."""
import pytest


def test_f5_tts_import():
    from livekit_agent.f5_tts_plugin import F5TTSPlugin
    assert F5TTSPlugin is not None


def test_f5_tts_constructor():
    from livekit_agent.f5_tts_plugin import F5TTSPlugin
    tts_inst = F5TTSPlugin(endpoint="http://localhost:18003")
    assert tts_inst.model == "Eempostor/F5-TTS-INDO-FINETUNE-V2"
    assert tts_inst.provider == "local-gx10"
    caps = tts_inst.capabilities
    assert caps.streaming is False
