import os
import platform

from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_n_gpu_layers() -> int:
    if platform.system() == "Darwin":
        return -1
    try:
        import torch
        return -1 if torch.cuda.is_available() else 0
    except ImportError:
        return 0


class Settings(BaseSettings):
    gguf_model_path: str = "./models/qwen3-1.7b-q4_k_m.gguf"
    fallback_model_path: str = "./models/qwen3-1.7b-q4_k_m.gguf"
    chroma_path: str = "./chroma_db"
    pdf_dir: str = "./documents"
    top_k: int = 5
    temperature: float = 0.3
    max_new_tokens: int = 512
    host: str = "0.0.0.0"
    port: int = 8000
    n_threads: int = max(1, (os.cpu_count() or 4) - 1)
    n_gpu_layers: int = _default_n_gpu_layers()
    max_history_turns: int = 6
    cache_ttl: int = 300
    api_key: str = ""
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    rate_limit: str = "10/minute"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
