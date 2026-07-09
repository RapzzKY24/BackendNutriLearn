from app.core.prompt import build_system_prompt, build_user_prompt


def test_system_prompt():
    prompt = build_system_prompt()
    assert "NutriAI" in prompt
    assert "Permenkes" in prompt


def test_user_prompt_with_context():
    result = build_user_prompt("konteks halaman 1", "apa itu gizi?")
    assert "konteks halaman 1" in result
    assert "apa itu gizi?" in result
    assert "HARUS cantumkan" in result


def test_user_prompt_without_context():
    result = build_user_prompt("", "pertanyaan saja")
    assert result == "pertanyaan saja"
