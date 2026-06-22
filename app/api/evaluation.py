from fastapi import APIRouter, HTTPException
from app.models.request import EvalRequest
from app.models.response import EvalResponse
from app.services.llm_service import llm_service
from app.core.logger import logger

router = APIRouter()

SYSTEM_PROMPT_NO_CONTEXT = (
    "Anda adalah asisten ahli gizi Indonesia yang bernama NutriAI.\n\n"
    "Jawab pertanyaan seputar gizi, kesehatan, dan pedoman gizi "
    "berdasarkan pengetahuan yang Anda miliki."
)


@router.post("/evaluate", response_model=EvalResponse)
async def evaluate_endpoint(req: EvalRequest):
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_NO_CONTEXT},
            {"role": "user", "content": req.question},
        ]
        answer, latency = llm_service.generate_with_latency(messages)
        return EvalResponse(answer=answer, latency=latency)
    except Exception as e:
        logger.error(f"Evaluate error: {e}")
        raise HTTPException(status_code=500, detail="Unable to process request")
