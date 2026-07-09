import os
from app.core.config import Settings


def test_settings_defaults():
    s = Settings()
    assert s.gguf_model_path == "./models/qwen3-4b-q4_k_m.gguf"
    assert s.fallback_model_path == "./models/qwen3-1.7b-q4_k_m.gguf"
    assert s.top_k == 5
    assert s.temperature == 0.3
    assert s.max_new_tokens == 512
    assert s.n_threads >= 1
    assert s.max_history_turns == 6
    assert s.cache_ttl == 300
    assert s.rate_limit == "10/minute"


def test_settings_env_override():
    os.environ["GGUF_MODEL_PATH"] = "/tmp/test.gguf"
    os.environ["API_KEY"] = "test-key-123"
    s = Settings()
    assert s.gguf_model_path == "/tmp/test.gguf"
    assert s.api_key == "test-key-123"
    del os.environ["GGUF_MODEL_PATH"]
    del os.environ["API_KEY"]
