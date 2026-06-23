import re
import numpy as np
from app.services.embedding_service import embedding_service
from app.core.logger import logger
from app.models.response import RAGEvalMetrics

THRESHOLD_RELEVANCE = 0.2
THRESHOLD_FAITHFUL = 0.4


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


class EvalRAGService:
    def evaluate(
        self,
        question: str,
        answer: str,
        context_docs: list[str],
        context_text_raw: str,
    ) -> RAGEvalMetrics:
        context_text = " ".join(context_docs)
        context_sentences = _split_sentences(context_text)
        answer_sentences = _split_sentences(answer)

        query_emb = embedding_service.embed([question])[0]

        # --- Context Relevance ---
        if context_sentences:
            ctx_embeds = embedding_service.embed(context_sentences)
            ctx_sims = [_cosine_similarity(query_emb, ce) for ce in ctx_embeds]
            relevant_count = sum(1 for s in ctx_sims if s > THRESHOLD_RELEVANCE)
            total_context = len(context_sentences)
            context_relevance = relevant_count / total_context
        else:
            relevant_count = 0
            total_context = 0
            context_relevance = 0.0

        # --- Answer Faithfulness ---
        if answer_sentences and context_sentences:
            ans_embeds = embedding_service.embed(answer_sentences)
            faithful_count = 0
            for ae in ans_embeds:
                max_sim = max(_cosine_similarity(ae, ce) for ce in ctx_embeds)
                if max_sim > THRESHOLD_FAITHFUL:
                    faithful_count += 1
            total_claims = len(answer_sentences)
            faithfulness = faithful_count / total_claims
        else:
            faithful_count = 0
            total_claims = 0
            faithfulness = 0.0

        # --- Answer Relevance ---
        if answer.strip():
            ans_emb = embedding_service.embed([answer])[0]
            answer_relevance = _cosine_similarity(query_emb, ans_emb)
        else:
            answer_relevance = 0.0

        answer_relevance = round(max(0.0, min(1.0, answer_relevance)), 2)

        # --- Analysis ---
        analysis_parts = []
        if context_relevance >= 0.7:
            analysis_parts.append("Konteks yang diberikan sangat relevan dengan pertanyaan.")
        elif context_relevance >= 0.4:
            analysis_parts.append("Konteks cukup relevan dengan pertanyaan.")
        else:
            analysis_parts.append("Konteks kurang relevan dengan pertanyaan.")

        if faithfulness >= 0.7:
            analysis_parts.append("Jawaban sangat setia pada konteks yang diberikan.")
        elif faithfulness >= 0.4:
            analysis_parts.append("Jawaban cukup setia pada konteks.")
        else:
            analysis_parts.append("Jawaban kurang setia pada konteks.")

        if answer_relevance >= 0.7:
            analysis_parts.append("Jawaban sangat relevan dengan pertanyaan.")
        elif answer_relevance >= 0.4:
            analysis_parts.append("Jawaban cukup relevan dengan pertanyaan.")
        else:
            analysis_parts.append("Jawaban kurang relevan dengan pertanyaan.")

        analysis = " ".join(analysis_parts)

        logger.info(
            f"Eval done — context_relevance={context_relevance:.2f} "
            f"({relevant_count}/{total_context}), "
            f"faithfulness={faithfulness:.2f} "
            f"({faithful_count}/{total_claims}), "
            f"answer_relevance={answer_relevance:.2f}"
        )

        return RAGEvalMetrics(
            context_relevance=round(context_relevance, 2),
            answer_faithfulness=round(faithfulness, 2),
            answer_relevance=answer_relevance,
            total_claims=total_claims,
            faithful_claims=faithful_count,
            relevant_sentences=relevant_count,
            total_sentences=total_context,
            analysis=analysis,
        )


eval_rag_service = EvalRAGService()
