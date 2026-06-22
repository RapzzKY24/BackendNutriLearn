import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.core.config import settings
from app.core.logger import logger

MODEL_PATHS = {
    "qwen25": "Qwen/Qwen2.5-0.5B-Instruct",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


class LLMService:
    def __init__(self):
        self.model_name = settings.model_name
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def _load_model(self):
        model_path = MODEL_PATHS.get(self.model_name)
        if not model_path:
            raise ValueError(f"Model '{self.model_name}' tidak dikenal. Pilihan: {list(MODEL_PATHS.keys())}")

        logger.info(f"Loading model: {model_path} on {DEVICE.upper()}...")
        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=settings.hf_token or None,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=DTYPE,
            device_map=DEVICE,
            token=settings.hf_token or None,
            trust_remote_code=True,
        )

        elapsed = time.time() - t0
        logger.info(f"Model loaded in {elapsed:.2f}s")
        self._loaded = True

    def generate(self, messages: list[dict]) -> str:
        if not self._loaded:
            self._load_model()

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        output = generated_ids[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(output, skip_special_tokens=True)
        return response.strip()

    def generate_with_latency(self, messages: list[dict]) -> tuple[str, float]:
        t0 = time.time()
        answer = self.generate(messages)
        latency = time.time() - t0
        return answer, round(latency, 2)

    @property
    def is_loaded(self) -> bool:
        return self._loaded


llm_service = LLMService()
