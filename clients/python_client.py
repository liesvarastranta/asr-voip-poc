"""Example downstream consumer: stream WAV audio, receive SSE transcripts."""
import asyncio
import json
import sys
from httpx import AsyncClient, Timeout


async def stream_audio(file_path: str, url: str = "http://localhost:8000/v1/asr/stream"):
    import soundfile as sf
    import numpy as np

    audio, sr = sf.read(file_path, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    pcm = audio.tobytes()

    chunk_duration_s = 0.5
    chunk_size = int(sr * 2 * chunk_duration_s)

    async with AsyncClient(timeout=Timeout(None)) as client:
        async with client.stream(
            "POST", url,
            content=iter_chunks(pcm, chunk_size),
            headers={
                "Content-Type": "application/octet-stream",
                "X-Audio-Sample-Rate": str(sr),
                "X-Audio-Channels": "1",
                "X-Audio-Format": "s16le",
                "X-Language": "id",
            },
        ) as resp:
            event_type = None
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event_type = line.split(":", 1)[1].strip()
                    print(f"\n[{event_type}]", end="")
                elif line.startswith("data:"):
                    data_str = line.split(":", 1)[1].strip()
                    try:
                        data = json.loads(data_str)
                        if event_type in ("partial", "final"):
                            print(f" {data.get('text', '')}", end="", flush=True)
                        elif event_type == "error":
                            print(f" ERROR: {data}", end="")
                        else:
                            print(f" {data}", end="")
                    except json.JSONDecodeError:
                        pass
            print()
            print(f"\nDone. Status: {resp.status_code}")


def iter_chunks(data: bytes, size: int):
    for i in range(0, len(data), size):
        yield data[i:i + size]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio_file.wav>")
        sys.exit(1)
    asyncio.run(stream_audio(sys.argv[1]))
