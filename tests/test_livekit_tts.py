"""Tests for ChatterboxTTSPlugin (Edge TTS default + local Chatterbox)."""
import pytest


def test_chatterbox_tts_import():
    from livekit_agent.chatterbox_tts_plugin import ChatterboxTTSPlugin
    assert ChatterboxTTSPlugin is not None


def test_tts_default_uses_edge():
    """No endpoint → default to Edge TTS (cloud, free, Indonesian)."""
    from livekit_agent.chatterbox_tts_plugin import ChatterboxTTSPlugin
    tts_inst = ChatterboxTTSPlugin()
    assert tts_inst._is_local is False
    assert tts_inst.provider == "edge"
    assert tts_inst.model == "edge-tts"
    assert tts_inst._voice == "id-ID-ArdiNeural"


def test_tts_local_mode():
    """TTS_ENDPOINT set to non-cloud URL → local Chatterbox mode."""
    from livekit_agent.chatterbox_tts_plugin import ChatterboxTTSPlugin
    tts_inst = ChatterboxTTSPlugin(endpoint="http://localhost:18003")
    assert tts_inst._is_local is True
    assert tts_inst.provider == "local-wsl2"
    assert tts_inst.model == "grandhigh/Chatterbox-TTS-Indonesian"


def test_tts_custom_voice():
    """Custom voice is honored (e.g. female Gadis)."""
    from livekit_agent.chatterbox_tts_plugin import ChatterboxTTSPlugin
    tts_inst = ChatterboxTTSPlugin(voice="id-ID-GadisNeural")
    assert tts_inst._voice == "id-ID-GadisNeural"
