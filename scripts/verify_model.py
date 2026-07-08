"""Phase 1 verification: model loads + offline transcribe works."""
import asyncio
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asr_service.engines.qwen_asr import Qwen3ASREngine
from asr_service.config import settings


async def main():
    print(f"Loading {settings.model_id} on {settings.device}...")
    engine = Qwen3ASREngine(
        model_id=settings.model_id,
        device=settings.device,
        dtype=settings.dtype,
    )
    await engine.load()
    print(f"Model loaded: {engine.is_loaded}")
    assert engine.is_loaded, "model failed to load"

    sample = os.getenv(
        "SAMPLE_AUDIO",
        "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
    )
    print(f"Transcribing {sample}...")
    result = await engine.transcribe_file(sample, language="id")
    print(f"Text: {result['text']}")
    print(f"Duration: {result['duration_ms']}ms, Processing: {result['processing_ms']}ms")
    rtf = result["processing_ms"] / result["duration_ms"]
    print(f"RTF: {rtf:.3f} ({'PASS' if rtf < 1.0 else 'WARN'})")


if __name__ == "__main__":
    asyncio.run(main())
