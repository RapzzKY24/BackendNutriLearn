from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    hf_token: str = ""
    model_name: Literal["qwen3"] = "qwen3"
    chroma_path: str = "./chroma_db"
    pdf_path: str = "./documents/Permenkes Nomor 41 Tahun 2014.pdf"
    top_k: int = 5
    temperature: float = 0.3
    max_new_tokens: int = 512
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
