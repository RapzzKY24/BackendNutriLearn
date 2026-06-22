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


class ErrorResponse(BaseModel):
    success: bool = False
    message: str = "Unable to process request"
