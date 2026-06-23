from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Pertanyaan user")


class BMIRequest(BaseModel):
    weight: float = Field(..., gt=0, lt=500, description="Berat badan dalam kg")
    height: float = Field(..., gt=0, lt=300, description="Tinggi badan dalam cm")


class EvalRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Pertanyaan untuk evaluasi")
    model: str = Field(..., description="Nama model untuk evaluasi")


class EvalRAGRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Pertanyaan untuk evaluasi RAG")
    answer: str = Field("", description="Jawaban untuk dievaluasi. Kosongkan untuk auto-generate")
