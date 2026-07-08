from fastapi import APIRouter, Request
from ..api.schemas import HealthResponse, ModelsResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    engine = request.app.state.engine
    return HealthResponse(
        status="ok" if engine.is_loaded else "degraded",
        model_loaded=engine.is_loaded,
        model=getattr(engine, "model_id", None),
        device=getattr(engine, "device", None),
    )


@router.get("/v1/models", response_model=ModelsResponse)
async def models(request: Request):
    engine = request.app.state.engine
    return ModelsResponse(models=[{
        "id": getattr(engine, "model_id", "unknown"),
        "device": getattr(engine, "device", "unknown"),
        "dtype": getattr(engine, "dtype_str", "unknown"),
        "status": "loaded" if engine.is_loaded else "loading",
    }])
