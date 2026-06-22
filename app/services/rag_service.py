from app.services.llm_service import llm_service
from app.services.retriever_service import retriever_service
from app.core.logger import logger
from app.models.response import Source


class RAGService:
    def ask(self, question: str) -> tuple[str, list[Source]]:
        if retriever_service.is_ready:
            logger.info("RAG: retrieving context from ChromaDB...")
            results = retriever_service.retrieve(question)
            context_parts = []
            seen_pages = set()
            for doc, meta in results:
                page = meta.get("page")
                if page:
                    seen_pages.add(page)
                    context_parts.append(f"[Halaman {page}]\n{doc}")
                else:
                    context_parts.append(doc)

            context = "\n\n---\n\n".join(context_parts) if context_parts else ""
            sources = [Source(page=p) for p in sorted(seen_pages)]
            logger.info(f"RAG: retrieved {len(context_parts)} chunks from pages {sorted(seen_pages)}")
        else:
            context = ""
            sources = []

        system_prompt = (
            "Anda adalah asisten ahli gizi Indonesia yang bernama NutriAI.\n\n"
            "Anda menjawab pertanyaan seputar gizi, kesehatan, dan pedoman gizi "
            "berdasarkan dokumen Peraturan Menteri Kesehatan (Permenkes) "
            "tentang pedoman gizi seimbang.\n\n"
            "ATURAN:\n"
            "1. Jawab hanya berdasarkan konteks yang diberikan.\n"
            "2. Gunakan bahasa Indonesia yang baik dan benar.\n"
            "3. Berikan jawaban secara naratif dan informatif.\n"
            "4. Setiap informasi yang kamu ambil dari konteks, sebutkan nomor halaman "
            "sumbernya di akhir kalimat, contoh: (Halaman 15).\n"
            "5. Jika informasi tidak tersedia dalam konteks, katakan bahwa "
            "informasi tidak ditemukan dalam dokumen Permenkes.\n"
            "6. JANGAN membuat informasi baru di luar konteks yang diberikan.\n"
            "7. JANGAN menggunakan pengetahuan umum atau pengetahuan luar.\n"
            "8. Jika ditanya di luar topik gizi, arahkan kembali ke topik gizi seimbang."
        )

        if context:
            system_prompt += f"\n\nContext:\n{context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        answer = llm_service.generate(messages)
        return answer, sources


rag_service = RAGService()
