import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.api import chat, health, bmi, evaluation, benchmark
from app.core.config import settings
from app.core.logger import logger
from app.services.llm_service import llm_service
from app.services.embedding_service import embedding_service



limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.rate_limit],
    enabled=bool(settings.rate_limit),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting NutriAI RAG Service...")
    if not settings.api_key:
        logger.warning("API_KEY is not set! Authentication is disabled.")
    if settings.gguf_model_path:
        try:
            await llm_service.ensure_loaded()
        except Exception as e:
            logger.error(f"Failed to load LLM at startup: {e}")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, embedding_service._ensure_loaded)
        logger.info("Embedding model pre-loaded at startup")
    except Exception as e:
        logger.error(f"Failed to load embedding model at startup: {e}")
    yield
    logger.info("Shutting down NutriAI RAG Service...")


app = FastAPI(
    title="NutriAI RAG Service",
    description="Backend RAG untuk NutriAI — Gizi & Kesehatan berdasarkan Permenkes",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

cors_origins = [
    o.strip()
    for o in settings.cors_origins.split(",")
    if o.strip()
] or ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    skip_paths = ["/docs", "/redoc", "/openapi.json", "/health", "/"]
    if request.url.path not in skip_paths:
        auth_header = request.headers.get("X-API-Key")
        if not settings.api_key:
            return JSONResponse(
                status_code=503,
                content={"detail": "Server not configured: API_KEY not set"},
            )
        if not auth_header or auth_header != settings.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
    return await call_next(request)


app.include_router(chat.router, tags=["Chat"])
app.include_router(health.router, tags=["Health"])
app.include_router(bmi.router, tags=["BMI"])
app.include_router(evaluation.router, tags=["Evaluation"])
app.include_router(benchmark.router)


@app.get("/")
async def root():
    return {
        "message": "NutriAI RAG Service is running",
        "version": "2.0.0",
        "docs": "/docs",
    }



