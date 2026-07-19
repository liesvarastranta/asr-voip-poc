"""Chatterbox-TTS Indonesian voice synthesis service."""
import asyncio
import io
import os
import torch
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
import soundfile as sf
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

app = FastAPI(title="Chatterbox-TTS Indonesian", version="0.1.0")

_model = None
MODEL_REPO = "grandhigh/Chatterbox-TTS-Indonesian"
CKPT_FILE = "t3_cfg.safetensors"


def get_model():
    global _model
    if _model is None:
        from chatterbox.tts import ChatterboxTTS
        _model = ChatterboxTTS.from_pretrained(device="cuda")
        ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename=CKPT_FILE)
        _model.t3.load_state_dict(load_file(ckpt_path, device="cpu"))
        torch.cuda.empty_cache()
    return _model


@app.get("/health")
async def health():
    try:
        get_model()
        return {"status": "ok", "model": MODEL_REPO}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/tts")
async def synthesize(
    text: str = Query(..., description="Indonesian text to synthesize"),
):
    model = get_model()
    # ponytail: run blocking generate in thread pool to not block event loop
    wav = await asyncio.to_thread(model.generate, text)
    if hasattr(wav, "cpu"):
        wav = wav.cpu().numpy()
    wav = wav.squeeze().astype("float32")
    buf = io.BytesIO()
    sf.write(buf, wav, model.sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=speech.wav"},
    )


@app.get("/")
async def root():
    return {"service": "Chatterbox-TTS Indonesian", "port": int(os.getenv("PORT", "18003"))}
