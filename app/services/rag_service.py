from app.services.llm_service import llm_service
from app.services.retriever_service import retriever_service
from app.core.logger import logger
from app.models.response import Source


class RAGService:
    async def ask(self, question: str) -> tuple[str, list[Source]]:
        context, sources = await self._retrieve_context(question)

        system_prompt = (
            "Anda adalah NutriAI, asisten ahli gizi Indonesia yang menjawab "
            "berdasarkan dokumen Permenkes tentang pedoman gizi seimbang.\n\n"
            "PENTING: Anda HARUS menjawab dalam Bahasa Indonesia. "
            "DILARANG menggunakan bahasa Inggris.\n\n"
            "Contoh:\n"
            "Pertanyaan: apa itu gizi seimbang?\n"
            "Jawaban: Gizi seimbang adalah susunan makanan sehari-hari yang "
            "mengandung zat gizi dalam jenis dan jumlah yang sesuai dengan "
            "kebutuhan tubuh.\n\n"
            "Jangan gunakan tag <think> atau proses berpikir apapun.\n"
            "Jawab langsung."
        )

        if context:
            user_prompt = (
                f"Konteks (setiap bagian memiliki label halaman):\n{context}\n\n"
                f"Pertanyaan: {question}\n\n"
                f"INSTRUKSI PENTING:\n"
                f"- Jawab berdasarkan konteks di atas.\n"
                f"- Jawab SELALU dalam Bahasa Indonesia.\n"
                f"- JANGAN sebutkan item yang sama lebih dari sekali.\n"
                f"- Jika ada istilah asing (Inggris) di konteks, terjemahkan ke Bahasa Indonesia.\n"
                f"- Gunakan istilah Indonesia: 'susu' bukan 'dairy', 'daging' bukan 'meat', 'telur' bukan 'egg', 'ikan' bukan 'fish'.\n"
                f"- SETIAP informasi yang kamu sebutkan, HARUS cantumkan "
                f"nomor halaman sumbernya, contoh: (Halaman 15).\n"
                f"- Jangan gunakan pengetahuan di luar konteks.\n"
                f"- Jika tidak ada di konteks, katakan tidak ditemukan."
            )
        else:
            user_prompt = f"Jawab dalam Bahasa Indonesia.\nPertanyaan: {question}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        answer = await llm_service.generate(messages)
        answer = self._deduplicate_lines(answer)
        if sources:
            source_str = ", ".join(f"Halaman {s.page}" for s in sources)
            answer += f"\n\n---\n📖 **Sumber:** {source_str}"
        return answer, sources

    @staticmethod
    def _deduplicate_lines(text: str) -> str:
        import re
        lines = text.split("\n")
        seen = set()
        result = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                result.append(line)
                continue
            key = re.sub(r'\s*\(.*?\)\s*', '', stripped).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(line)
        return "\n".join(result)

    async def _retrieve_context(self, question: str) -> tuple[str, list[Source]]:
        if retriever_service.is_ready:
            logger.info("RAG: retrieving context from ChromaDB...")
            results = retriever_service.retrieve(question)
            context_parts = []
            seen_pages = set()
            for doc, meta in results:
                page = meta.get("page")
                if page:
                    if page not in seen_pages:
                        seen_pages.add(page)
                        context_parts.append(f"[Halaman {page}]\n{doc}")
                else:
                    context_parts.append(doc)

            context = "\n\n---\n\n".join(context_parts) if context_parts else ""
            sources = [Source(page=p) for p in sorted(seen_pages)]
            logger.info(f"RAG: retrieved {len(context_parts)} chunks from pages {sorted(seen_pages)}")
            return context, sources

        return "", []


rag_service = RAGService()
