import numpy as np
from asr_service.audio.resample import pcm_bytes_to_float32, float32_to_pcm_bytes
from asr_service.audio.vad import energy_vad

def test_pcm_to_float32():
    pcm = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16).tobytes()
    floats = pcm_bytes_to_float32(pcm)
    assert floats.shape == (5,)
    assert np.allclose(floats, [0.0, 0.5, -0.5, 1.0, -1.0], atol=0.01)

def test_float32_to_pcm():
    floats = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    pcm = float32_to_pcm_bytes(floats)
    result = np.frombuffer(pcm, dtype=np.int16)
    expected = np.array([0, 16383, -16383], dtype=np.int16)
    assert np.abs(result - expected).max() <= 1  # quantization tolerance

def test_pcm_roundtrip():
    original = np.array([0, 10000, -10000, 20000, -20000], dtype=np.int16)
    floats = pcm_bytes_to_float32(original.tobytes())
    pcm = float32_to_pcm_bytes(floats)
    result = np.frombuffer(pcm, dtype=np.int16)
    assert np.abs(result - original).max() <= 1

def test_vad_silence():
    silence = np.zeros(16000, dtype=np.float32)
    assert energy_vad(silence, threshold=0.01) is False

def test_vad_speech():
    t = np.linspace(0, 1, 16000, endpoint=False)
    speech = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    assert energy_vad(speech, threshold=0.01) is True

def test_vad_threshold():
    quiet = np.array([0.001] * 1000, dtype=np.float32)
    assert energy_vad(quiet, threshold=0.01) is False
