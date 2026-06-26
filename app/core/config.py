from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gguf_model_path: str = "./models/qwen3-1.7b-q4_k_m.gguf"
    chroma_path: str = "./chroma_db"
    pdf_path: str = "./documents/Permenkes Nomor 41 Tahun 2014.pdf"
    top_k: int = 5
    temperature: float = 0.3
    max_new_tokens: int = 512
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
