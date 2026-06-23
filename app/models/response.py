from pydantic import BaseModel
from typing import Optional


class Source(BaseModel):
    page: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []


class BMIResponse(BaseModel):
    bmi: float
    category: str


class EvalResponse(BaseModel):
    answer: str
    latency: float


class RAGEvalMetrics(BaseModel):
    context_relevance: float
    answer_faithfulness: float
    answer_relevance: float
    total_claims: int
    faithful_claims: int
    relevant_sentences: int
    total_sentences: int
    analysis: str


class EvalRAGResponse(BaseModel):
    question: str
    answer: str
    context_sources: list[Source]
    metrics: RAGEvalMetrics


class ErrorResponse(BaseModel):
    success: bool = False
    message: str = "Unable to process request"
