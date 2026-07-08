import numpy as np

def pcm_bytes_to_float32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

def float32_to_pcm_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()

def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    try:
        import torchaudio
        import torch
        wav = torch.from_numpy(audio).unsqueeze(0)
        resampled = torchaudio.functional.resample(wav, orig_sr, target_sr)
        return resampled.squeeze(0).numpy()
    except ImportError:
        pass
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)
