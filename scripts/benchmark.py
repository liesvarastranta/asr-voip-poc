"""Benchmark RTF, latency on GX10."""
import asyncio
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asr_service.engines.qwen_asr import Qwen3ASREngine
from asr_service.config import settings


async def benchmark_file(path: str):
    engine = Qwen3ASREngine(
        model_id=settings.model_id,
        device=settings.device,
        dtype=settings.dtype,
    )
    await engine.load()

    t0 = time.monotonic()
    result = await engine.transcribe_file(path, language="id")
    elapsed = time.monotonic() - t0

    duration_s = result["duration_ms"] / 1000
    rtf = elapsed / duration_s

    print(f"File: {path}")
    print(f"  Duration:   {duration_s:.2f}s")
    print(f"  Processing: {elapsed:.3f}s")
    print(f"  RTF:        {rtf:.3f} ({'PASS' if rtf < 1.0 else 'FAIL'})")
    print(f"  Text:       {result['text'][:80]}...")

    return rtf


async def main():
    files = os.getenv("BENCH_FILES", "").split(",")
    files = [f.strip() for f in files if f.strip()]
    if not files:
        print("Set BENCH_FILES env var (comma-separated audio paths)")
        return

    all_rtf = []
    for f in files:
        all_rtf.append(await benchmark_file(f))

    avg = sum(all_rtf) / len(all_rtf)
    print(f"\nAverage RTF: {avg:.3f}")
    print(f"Overall: {'PASS' if avg < 1.0 else 'FAIL'}")


if __name__ == "__main__":
    asyncio.run(main())
