from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat, health, bmi, evaluation
from app.core.config import settings

app = FastAPI(
    title="NutriAI RAG Service",
    description="Backend RAG untuk NutriAI — Gizi & Kesehatan berdasarkan Permenkes",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, tags=["Chat"])
app.include_router(health.router, tags=["Health"])
app.include_router(bmi.router, tags=["BMI"])
app.include_router(evaluation.router, tags=["Evaluation"])


@app.get("/")
async def root():
    return {"message": "NutriAI RAG Service is running", "docs": "/docs"}
