import pytest


@pytest.mark.asyncio
async def test_stream_basic(client):
    pcm = b"\x00\x00" * 32000
    async with client.stream(
        "POST", "/v1/asr/stream",
        content=pcm,
        headers={"Content-Type": "application/octet-stream"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = []
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
        assert "ready" in events
        assert "final" in events


@pytest.mark.asyncio
async def test_stream_headers(client):
    pcm = b"\x00\x00" * 3200
    async with client.stream(
        "POST", "/v1/asr/stream",
        content=pcm,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Language": "id",
            "X-Audio-Sample-Rate": "16000",
            "X-Audio-Channels": "1",
        },
    ) as resp:
        assert resp.status_code == 200
