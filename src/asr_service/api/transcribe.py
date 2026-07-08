import os
import tempfile
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from ..api.schemas import TranscribeResponse

router = APIRouter()

MAX_FILE_BYTES = 100 * 1024 * 1024


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("id"),
):
    engine = request.app.state.engine
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="model not loaded")

    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="file too large")

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        result = await engine.transcribe_file(tmp_path, language=language)
        return TranscribeResponse(**result)
    finally:
        os.unlink(tmp_path)
