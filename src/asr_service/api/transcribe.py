import os
import tempfile
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from ..api.schemas import TranscribeResponse

router = APIRouter()


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("id"),
):
    engine = request.app.state.engine
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="model not loaded")

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await engine.transcribe_file(tmp_path, language=language)
        return TranscribeResponse(**result)
    finally:
        os.unlink(tmp_path)
