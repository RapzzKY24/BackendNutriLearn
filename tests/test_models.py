from app.models.request import ChatRequest, BMIRequest
from app.models.response import ChatResponse, Source, BMIResponse


def test_chat_request():
    req = ChatRequest(question="Apa itu gizi?")
    assert req.question == "Apa itu gizi?"
    assert req.session_id == ""


def test_chat_request_with_session():
    req = ChatRequest(question="Apa itu gizi?", session_id="abc123")
    assert req.session_id == "abc123"


def test_chat_response():
    resp = ChatResponse(
        answer="Gizi adalah...",
        sources=[Source(page=15), Source(page=16)],
    )
    assert len(resp.sources) == 2
    assert resp.sources[0].page == 15


def test_bmi_request():
    req = BMIRequest(weight=65, height=170)
    assert req.weight == 65
    assert req.height == 170


def test_bmi_response():
    resp = BMIResponse(bmi=22.5, category="Normal")
    assert resp.bmi == 22.5
    assert resp.category == "Normal"
