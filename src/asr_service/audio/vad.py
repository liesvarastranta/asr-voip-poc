import numpy as np

def energy_vad(audio: np.ndarray, threshold: float = 0.01) -> bool:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return rms > threshold

def detect_speech_segments(
    audio: np.ndarray, sample_rate: int = 16000,
    frame_ms: int = 30, threshold: float = 0.01,
) -> list[tuple[float, float]]:
    frame_size = int(sample_rate * frame_ms / 1000)
    segments = []
    in_speech = False
    start = 0.0
    for i in range(0, len(audio), frame_size):
        frame = audio[i:i + frame_size]
        if len(frame) < frame_size:
            break
        is_speech = energy_vad(frame, threshold)
        if is_speech and not in_speech:
            in_speech = True
            start = i / sample_rate
        elif not is_speech and in_speech:
            in_speech = False
            segments.append((start, i / sample_rate))
    if in_speech:
        segments.append((start, len(audio) / sample_rate))
    return segments
