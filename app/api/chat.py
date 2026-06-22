from fastapi import APIRouter, HTTPException
from app.models.request import ChatRequest
from app.models.response import ChatResponse, ErrorResponse
from app.services.rag_service import rag_service
from app.services.llm_service import llm_service
from app.core.logger import logger

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def chat_endpoint(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer, sources = rag_service.ask(req.question)
        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Unable to process request")
