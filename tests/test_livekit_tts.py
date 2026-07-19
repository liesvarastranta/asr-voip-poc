"""Tests for ChatterboxTTSPlugin."""
import pytest


def test_chatterbox_tts_import():
    from livekit_agent.chatterbox_tts_plugin import ChatterboxTTSPlugin
    assert ChatterboxTTSPlugin is not None


def test_chatterbox_tts_constructor():
    from livekit_agent.chatterbox_tts_plugin import ChatterboxTTSPlugin
    tts_inst = ChatterboxTTSPlugin(endpoint="http://localhost:18003")
    assert tts_inst.model == "grandhigh/Chatterbox-TTS-Indonesian"
    assert tts_inst.provider == "local-wsl2"
    caps = tts_inst.capabilities
    assert caps.streaming is False
