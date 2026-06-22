from fastapi import APIRouter, HTTPException
from app.models.request import BMIRequest
from app.models.response import BMIResponse

router = APIRouter()


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100
    return round(weight_kg / (height_m * height_m), 1)


def get_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Kekurangan Berat Badan"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Kelebihan Berat Badan"
    else:
        return "Obesitas"


@router.post("/bmi", response_model=BMIResponse)
async def bmi_endpoint(req: BMIRequest):
    if req.height <= 0:
        raise HTTPException(status_code=400, detail="Tinggi badan harus lebih dari 0")

    bmi = calculate_bmi(req.weight, req.height)
    category = get_category(bmi)

    return BMIResponse(bmi=bmi, category=category)
