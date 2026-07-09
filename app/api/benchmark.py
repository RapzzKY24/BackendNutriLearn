import json
import os
import time

from fastapi import APIRouter, HTTPException

from app.core.input_guard import guard_question
from app.core.logger import logger
from app.models.response import ErrorResponse
from app.services.eval_rag_service import eval_rag_service
from app.services.llm_service import llm_service
from app.services.rag_service import rag_service

router = APIRouter(tags=["Benchmark"])

MODELS_CONFIG_PATH = os.getenv(
    "MODELS_CONFIG_PATH",
    "./models/models_config.json",
)


def _load_models_config() -> list[dict]:
    if not os.path.exists(MODELS_CONFIG_PATH):
        return []
    with open(MODELS_CONFIG_PATH) as f:
        data = json.load(f)
    return data.get("models", [])


def _available_models() -> list[dict]:
    return [m for m in _load_models_config() if os.path.exists(m["path"])]


async def _run_single_model(model_cfg: dict, question: str) -> dict:
    model_name = model_cfg["name"]
    logger.info(f"Benchmarking model: {model_name}")

    await llm_service.swap_model(model_cfg["path"], model_name)

    t0 = time.time()

    answer, sources = await rag_service.ask(question)

    context_str, _ = await rag_service._retrieve_context(question)
    context_docs = context_str.split("\n\n---\n\n") if context_str else []

    metrics = eval_rag_service.evaluate(
        question=question,
        answer=answer,
        context_docs=context_docs,
        context_text_raw=context_str,
    )

    latency = round(time.time() - t0, 2)

    combined_score = round(
        metrics.context_relevance * 0.3
        + metrics.answer_faithfulness * 0.4
        + metrics.answer_relevance * 0.3,
        4,
    )

    logger.info(
        f"{model_name} done — "
        f"combined={combined_score}, "
        f"faithfulness={metrics.answer_faithfulness}, "
        f"latency={latency}s"
    )

    return {
        "model": model_name,
        "model_id": model_cfg["id"],
        "answer": answer,
        "latency": latency,
        "combined_score": combined_score,
        "metrics": {
            "context_relevance": metrics.context_relevance,
            "answer_faithfulness": metrics.answer_faithfulness,
            "answer_relevance": metrics.answer_relevance,
            "total_claims": metrics.total_claims,
            "faithful_claims": metrics.faithful_claims,
            "relevant_sentences": metrics.relevant_sentences,
            "total_sentences": metrics.total_sentences,
            "analysis": metrics.analysis,
        },
    }


@router.post("/ask")
async def ask_models(query: str):
    try:
        query = guard_question(query)
    except ValueError as e:
        return ErrorResponse(message=str(e))

    models = _available_models()
    if not models:
        return ErrorResponse(message="No models available. Run download_models.sh first.")

    all_results = []
    for m in models:
        try:
            result = await _run_single_model(m, query)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error with model {m['name']}: {e}")
            all_results.append({
                "model": m["name"],
                "model_id": m["id"],
                "answer": "",
                "latency": 0,
                "combined_score": 0,
                "metrics": None,
                "error": "Model failed to load or generate",
            })

    valid = [r for r in all_results if r["combined_score"] > 0]
    if not valid:
        return ErrorResponse(message="All models failed to run.")

    best = max(valid, key=lambda x: x["combined_score"])

    return {
        "best_model": best["model"],
        "answer": best["answer"],
        "combined_score": best["combined_score"],
        "latency": best["latency"],
        "metrics": best["metrics"],
    }


@router.post("/benchmark")
async def benchmark(query: str):
    try:
        query = guard_question(query)
    except ValueError as e:
        return ErrorResponse(message=str(e))

    models = _available_models()
    if not models:
        return ErrorResponse(message="No models available. Run download_models.sh first.")

    all_results = []
    for m in models:
        try:
            result = await _run_single_model(m, query)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error with model {m['name']}: {e}")
            all_results.append({
                "model": m["name"],
                "model_id": m["id"],
                "answer": "",
                "latency": 0,
                "combined_score": 0,
                "metrics": None,
                "error": "Model failed to load or generate",
            })

    valid = [r for r in all_results if r["combined_score"] > 0]
    if not valid:
        return ErrorResponse(message="All models failed to run.")

    best = max(valid, key=lambda x: x["combined_score"])

    return {
        "best_model": best["model"],
        "answer": best["answer"],
        "combined_score": best["combined_score"],
        "latency": best["latency"],
        "metrics": best["metrics"],
        "all_results": [
            {
                "model": r["model"],
                "combined_score": r["combined_score"],
                "latency": r["latency"],
                "metrics": r["metrics"],
                "error": r.get("error"),
            }
            for r in all_results
        ],
    }
