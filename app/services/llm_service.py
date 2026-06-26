import time
import re
from llama_cpp import Llama
from app.core.config import settings
from app.core.logger import logger


class LLMService:
    def __init__(self):
        self.model_path = settings.gguf_model_path
        self.model = None
        self._loaded = False

    def _load_model(self):
        logger.info(f"Loading GGUF model: {self.model_path}...")
        t0 = time.time()

        self.model = Llama(
            model_path=self.model_path,
            n_ctx=8192,
            n_gpu_layers=-1,
            n_threads=4,
            verbose=False,
        )

        elapsed = time.time() - t0
        logger.info(f"Model loaded in {elapsed:.2f}s")
        self._loaded = True

    def generate(self, messages: list[dict]) -> str:
        if not self._loaded:
            self._load_model()

        output = self.model.create_chat_completion(
            messages=messages,
            max_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=0.9,
        )
        response = output["choices"][0]["message"]["content"]
        response = self._strip_thinking(response)
        return response.strip()

    def _strip_thinking(self, text: str) -> str:
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def generate_with_latency(self, messages: list[dict]) -> tuple[str, float]:
        t0 = time.time()
        answer = self.generate(messages)
        latency = time.time() - t0
        return answer, round(latency, 2)

    @property
    def is_loaded(self) -> bool:
        return self._loaded


llm_service = LLMService()
