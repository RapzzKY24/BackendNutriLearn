#!/bin/bash
set -e

MODEL_URL="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
MODEL_FILE="/app/models/llama-3.2-3b-instruct-q4_k_m.gguf"
INGEST_MARKER="/app/chroma_db/.ingested"

echo "=== NutriAI Startup ==="

# ── Download model ──
if [ -f "$MODEL_FILE" ]; then
    echo "[OK] Model already exists: $(du -h "$MODEL_FILE" | cut -f1)"
else
    echo "[...] Downloading Llama-3.2-3B (~1.9GB)..."
    wget -q --show-progress -O "$MODEL_FILE" "$MODEL_URL"
    echo "[OK] Model downloaded"
fi

# ── Ingest documents ──
if [ -f "$INGEST_MARKER" ]; then
    echo "[OK] ChromaDB already ingested"
else
    echo "[...] Running document ingestion..."
    python /app/ingest.py
    touch "$INGEST_MARKER"
    echo "[OK] Ingestion complete"
fi

echo "=== Starting NutriAI Server ==="
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
