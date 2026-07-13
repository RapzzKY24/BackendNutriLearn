import asyncio
import gc
import os
import re
import time
from llama_cpp import Llama
from app.core.config import settings
from app.core.logger import logger


class LLMService:
    def __init__(self):
        self.model_path = settings.gguf_model_path
        self.fallback_path = settings.fallback_model_path
        self.model = None
        self._loaded = False
        self._loop = None
        self._model_name = ""

    @property
    def current_model_name(self) -> str:
        return self._model_name or os.path.basename(self.model_path)

    async def swap_model(self, path: str, name: str = ""):
        self._loop = asyncio.get_event_loop()
        logger.info(f"Swapping model to: {name or path}")
        await self._unload()
        gc.collect()
        await self._load_model(path)
        self._model_name = name

    async def _unload(self):
        if self.model is not None:
            logger.info("Unloading current model...")
            del self.model
            self.model = None
            self._loaded = False
            gc.collect()
            logger.info("Model unloaded")

    async def ensure_loaded(self):
        if self._loaded:
            return
        self._loop = asyncio.get_event_loop()
        await self._load_model()

    async def _load_model(self, path: str | None = None):
        target = path or self.model_path
        logger.info(f"Loading GGUF model: {target}...")
        t0 = time.time()

        def _load(p: str):
            return Llama(
                model_path=p,
                n_ctx=settings.n_ctx,
                batch_size=settings.batch_size,
                n_gpu_layers=settings.n_gpu_layers,
                n_threads=settings.n_threads,
                verbose=False,
            )

        try:
            self.model = await self._loop.run_in_executor(None, _load, target)
            self.model_path = target
            elapsed = time.time() - t0
            logger.info(f"Model loaded in {elapsed:.2f}s")
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed to load model from {target}: {e}")
            if path is None and self.fallback_path and os.path.exists(self.fallback_path):
                logger.warning(f"Falling back to: {self.fallback_path}")
                await self._load_model(self.fallback_path)
            else:
                raise RuntimeError(f"Cannot load any model: {e}")

    def _generate_sync(self, messages: list[dict]) -> str:
        output = self.model.create_chat_completion(
            messages=messages,
            max_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=0.9,
        )
        response = output["choices"][0]["message"]["content"]
        return self._strip_thinking(response).strip()

    async def generate(self, messages: list[dict]) -> str:
        if not self._loaded:
            await self.ensure_loaded()
        loop = self._loop or asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, self._generate_sync, messages),
            timeout=settings.llm_timeout,
        )

    async def generate_stream(self, messages: list[dict]):
        if not self._loaded:
            await self.ensure_loaded()
        loop = self._loop or asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run():
            try:
                stream = self.model.create_chat_completion(
                    messages=messages,
                    max_tokens=settings.max_new_tokens,
                    temperature=settings.temperature,
                    top_p=0.9,
                    stream=True,
                )
                buffer = ""
                in_think = False
                for chunk in stream:
                    delta = chunk["choices"][0]["delta"]
                    if "content" not in delta:
                        continue
                    content = delta["content"]
                    buffer += content
                    text, next_in_think, remainder = self._extract_visible_text(buffer, in_think)
                    if text is not None:
                        in_think = next_in_think
                        buffer = remainder
                        asyncio.run_coroutine_threadsafe(
                            queue.put(("token", text)), loop
                        ).result()
                    else:
                        in_think = next_in_think
                        buffer = remainder
                remaining = self._strip_thinking(buffer)
                if remaining:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("token", remaining)), loop
                    ).result()
                asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop).result()
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    queue.put(("error", str(e))), loop
                ).result()

        loop.run_in_executor(None, _run)

        while True:
            kind, payload = await asyncio.wait_for(queue.get(), timeout=settings.llm_timeout)
            if kind == "token":
                yield payload
            elif kind == "done":
                break
            elif kind == "error":
                raise Exception(payload)

    def _extract_visible_text(self, buffer: str, in_think: bool):
        if in_think:
            idx = buffer.find("</think>")
            if idx != -1:
                after = buffer[idx + 8:]
                if after:
                    return (after, False, "")
                return (None, False, "")
            return (None, True, buffer)

        idx = buffer.find("<think>")
        if idx != -1:
            before = buffer[:idx]
            after = buffer[idx + 7:]
            end_idx = after.find("</think>")
            if end_idx != -1:
                visible = before + after[end_idx + 8:]
                if visible:
                    return (visible, False, "")
                return (None, False, "")
            if before:
                return (before, True, after)
            return (None, True, after)

        if buffer:
            return (buffer, False, "")
        return (None, False, "")

    async def generate_with_latency(self, messages: list[dict]) -> tuple[str, float]:
        t0 = time.time()
        answer = await self.generate(messages)
        latency = time.time() - t0
        return answer, round(latency, 2)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
        return text.strip()

    @property
    def is_loaded(self) -> bool:
        return self._loaded


llm_service = LLMService()
