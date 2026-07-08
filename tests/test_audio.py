import numpy as np
from asr_service.audio.resample import pcm_bytes_to_float32, float32_to_pcm_bytes

def test_pcm_to_float32():
    pcm = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16).tobytes()
    floats = pcm_bytes_to_float32(pcm)
    assert floats.shape == (5,)
    assert np.allclose(floats, [0.0, 0.5, -0.5, 1.0, -1.0], atol=0.01)

def test_pcm_roundtrip():
    original = np.array([0, 10000, -10000, 20000, -20000], dtype=np.int16)
    floats = pcm_bytes_to_float32(original.tobytes())
    pcm = float32_to_pcm_bytes(floats)
    result = np.frombuffer(pcm, dtype=np.int16)
    assert np.abs(result - original).max() <= 1
