from fastapi import APIRouter, HTTPException
from app.models.request import EvalRequest, EvalRAGRequest
from app.models.response import EvalResponse, EvalRAGResponse, Source
from app.services.llm_service import llm_service
from app.services.rag_service import rag_service
from app.services.retriever_service import retriever_service
from app.services.eval_rag_service import eval_rag_service
from app.core.input_guard import guard_question
from app.core.logger import logger

router = APIRouter()

SYSTEM_PROMPT_NO_CONTEXT = (
    "Anda adalah asisten ahli gizi Indonesia yang bernama NutriAI.\n\n"
    "Jawab pertanyaan seputar gizi, kesehatan, dan pedoman gizi "
    "berdasarkan pengetahuan yang Anda miliki.\n"
    "Gunakan Bahasa Indonesia yang baik dan benar.\n"
    "Jangan gunakan tag <think> atau proses berpikir apapun.\n"
    "Jawab langsung tanpa analisis."
)


@router.post("/evaluate", response_model=EvalResponse)
async def evaluate_endpoint(req: EvalRequest):
    try:
        guard_question(req.question)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_NO_CONTEXT},
            {"role": "user", "content": req.question},
        ]
        answer, latency = await llm_service.generate_with_latency(messages)
        return EvalResponse(answer=answer, latency=latency)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Evaluate error: {e}")
        raise HTTPException(status_code=500, detail="Unable to process request")


@router.post("/evaluate/rag", response_model=EvalRAGResponse)
async def evaluate_rag_endpoint(req: EvalRAGRequest):
    try:
        guard_question(req.question)

        if not retriever_service.is_ready:
            raise HTTPException(status_code=400, detail="ChromaDB belum di-populate. Jalankan ingest.py dulu.")

        results = retriever_service.retrieve(req.question)
        context_docs = [doc for doc, _ in results]
        seen_pages = sorted({meta.get("page") for _, meta in results if meta.get("page")})
        sources = [Source(page=p) for p in seen_pages]
        context_text = "\n\n---\n\n".join(context_docs)

        answer = req.answer.strip() if req.answer.strip() else None
        if not answer:
            answer, _ = await rag_service.ask(req.question)
            answer = answer.split("\n\n---\n📖")[0]

        metrics = eval_rag_service.evaluate(
            question=req.question,
            answer=answer,
            context_docs=context_docs,
            context_text_raw=context_text,
        )

        return EvalRAGResponse(
            question=req.question,
            answer=answer,
            context_sources=sources,
            metrics=metrics,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Evaluate RAG error: {e}")
        raise HTTPException(status_code=500, detail="Unable to process request")
