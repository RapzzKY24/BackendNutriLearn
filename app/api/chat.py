import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.models.request import ChatRequest
from app.models.response import ChatResponse, ErrorResponse
from app.services.llm_service import llm_service
from app.services.rag_service import rag_service
from app.core.input_guard import guard_question
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
        guard_question(req.question)
        answer, sources = await rag_service.ask(req.question)
        return ChatResponse(answer=answer, sources=sources)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Unable to process request")


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        guard_question(req.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    async def event_stream():
        try:
            context, sources = await rag_service._retrieve_context(req.question)

            system_prompt = (
                "Anda adalah NutriAI, asisten ahli gizi Indonesia yang menjawab "
                "berdasarkan dokumen Permenkes tentang pedoman gizi seimbang.\n\n"
                "PENTING: Anda HARUS menjawab dalam Bahasa Indonesia. "
                "DILARANG menggunakan bahasa Inggris.\n\n"
                "Contoh:\n"
                "Pertanyaan: apa itu gizi seimbang?\n"
                "Jawaban: Gizi seimbang adalah susunan makanan sehari-hari yang "
                "mengandung zat gizi dalam jenis dan jumlah yang sesuai dengan "
                "kebutuhan tubuh.\n\n"
                "Jangan gunakan tag <think> atau proses berpikir apapun.\n"
                "Jawab langsung."
            )

            if context:
                user_prompt = (
                    f"Konteks (setiap bagian memiliki label halaman):\n{context}\n\n"
                    f"Pertanyaan: {req.question}\n\n"
                    f"INSTRUKSI PENTING:\n"
                    f"- Jawab berdasarkan konteks di atas.\n"
                    f"- Jawab SELALU dalam Bahasa Indonesia.\n"
                    f"- JANGAN sebutkan item yang sama lebih dari sekali.\n"
                    f"- Jika ada istilah asing (Inggris) di konteks, terjemahkan ke Bahasa Indonesia.\n"
                    f"- Gunakan istilah Indonesia: 'susu' bukan 'dairy', 'daging' bukan 'meat', 'telur' bukan 'egg', 'ikan' bukan 'fish'.\n"
                    f"- SETIAP informasi yang kamu sebutkan, HARUS cantumkan "
                    f"nomor halaman sumbernya, contoh: (Halaman 15).\n"
                    f"- Jangan gunakan pengetahuan di luar konteks.\n"
                    f"- Jika tidak ada di konteks, katakan tidak ditemukan."
                )
            else:
                user_prompt = f"Jawab dalam Bahasa Indonesia.\nPertanyaan: {req.question}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            async for token in llm_service.generate_stream(messages):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            if sources:
                source_str = ", ".join(f"Halaman {s.page}" for s in sources)
                yield f"data: {json.dumps({'type': 'sources', 'content': source_str})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
