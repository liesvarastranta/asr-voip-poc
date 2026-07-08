import json
from fastapi import APIRouter, Request, Header
from sse_starlette.sse import EventSourceResponse
from ..config import settings

router = APIRouter()


@router.post("/stream")
async def transcribe_stream(
    request: Request,
    x_audio_sample_rate: int = Header(16000),
    x_audio_channels: int = Header(1),
    x_audio_format: str = Header("s16le"),
    x_language: str = Header("id"),
):
    engine = request.app.state.engine
    bytes_per_sample = 2
    chunk_bytes = x_audio_sample_rate * bytes_per_sample * (settings.chunk_ms // 1000)

    async def event_generator():
        buffer = bytearray()
        last_infer_len = 0

        yield {"event": "ready", "data": json.dumps({})}

        async for chunk in request.stream():
            buffer.extend(chunk)
            t_ms = len(buffer) * 1000 // (x_audio_sample_rate * bytes_per_sample)

            if len(buffer) - last_infer_len >= chunk_bytes:
                try:
                    text = await engine.infer_chunk(bytes(buffer), is_final=False)
                    last_infer_len = len(buffer)
                    if text:
                        yield {
                            "event": "partial",
                            "data": json.dumps({"text": text, "is_final": False, "t_ms": t_ms}),
                        }
                except Exception as e:
                    yield {
                        "event": "error",
                        "data": json.dumps({"code": "INTERNAL_ERROR", "message": str(e)}),
                    }
                    return

        if buffer:
            try:
                text = await engine.infer_chunk(bytes(buffer), is_final=True)
                t_ms = len(buffer) * 1000 // (x_audio_sample_rate * bytes_per_sample)
                yield {
                    "event": "final",
                    "data": json.dumps({
                        "text": text, "is_final": True,
                        "t_ms": t_ms, "language": x_language,
                    }),
                }
            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps({"code": "INTERNAL_ERROR", "message": str(e)}),
                }

    return EventSourceResponse(event_generator())
