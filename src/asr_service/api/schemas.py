from pydantic import BaseModel

class TranscribeResponse(BaseModel):
    text: str
    language: str
    duration_ms: int
    processing_ms: int | None = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model: str | None = None
    device: str | None = None

class ModelsResponse(BaseModel):
    models: list[dict]

class SSEPartial(BaseModel):
    text: str
    is_final: bool = False
    t_ms: int

class SSEFinal(BaseModel):
    text: str
    is_final: bool = True
    t_ms: int
    language: str

class SSEError(BaseModel):
    code: str
    message: str

class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
