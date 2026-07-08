"""F5-TTS Indonesian voice synthesis service."""
import io
import os
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
import soundfile as sf

app = FastAPI(title="F5-TTS Indonesian TTS Service", version="0.1.0")

# Lazy-loaded model
_f5tts = None
_ref_audio_path = os.getenv("REF_AUDIO_PATH", "ref_audio/ref.wav")
_ref_text = os.getenv("REF_TEXT", "Selamat pagi, salam sejahtera bagi kita sekalian.")
_ckpt_dir = os.getenv("CKPT_DIR", "/models/Eempostor/F5-TTS-INDO-FINETUNE-V2")

def get_model():
    global _f5tts
    if _f5tts is None:
        from f5_tts.api import F5TTS
        _f5tts = F5TTS(ckpt_file=_ckpt_dir)
    return _f5tts

@app.get("/health")
async def health():
    try:
        get_model()
        return {"status": "ok", "model": "Eempostor/F5-TTS-INDO-FINETUNE-V2"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/tts")
async def synthesize(text: str = Query(..., description="Indonesian text to synthesize")):
    f5tts = get_model()
    wav, sr, _ = f5tts.infer(
        ref_file=_ref_audio_path,
        ref_text=_ref_text,
        gen_text=text,
        file_wave=None,
        seed=-1,
    )
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav",
                            headers={"Content-Disposition": "inline; filename=speech.wav"})

@app.get("/")
async def root():
    return {"service": "F5-TTS Indonesian", "port": int(os.getenv("PORT", "18003"))}
