from fastapi import APIRouter
from app.core.config import settings
from app.services.llm_service import llm_service

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model": llm_service.current_model_name,
        "model_loaded": llm_service.is_loaded,
    }
