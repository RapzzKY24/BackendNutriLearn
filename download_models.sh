#!/bin/bash
set -e

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

echo "========================================"
echo " Downloading all benchmark models"
echo " Total: ~24 GB"
echo "========================================"
echo ""

download() {
    local name="$1"
    local url="$2"
    local path="$MODELS_DIR/$name"

    if [ -f "$path" ]; then
        echo "[SKIP] $name already exists"
        return 0
    fi

    echo "[DOWNLOAD] $name ..."
    wget -O "$path" "$url"
    echo "[DONE] $name"
    echo ""
}

# 1 - Sudah ada: Qwen3-1.7B

# 2 - Qwen2.5-3B
download "qwen2.5-3b-instruct-q4_k_m.gguf" \
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

# 3 - Qwen3-4B
download "qwen3-4b-q4_k_m.gguf" \
    "https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen_Qwen3-4B-Q4_K_M.gguf"

# 4 - Llama-3.2-3B
download "llama-3.2-3b-instruct-q4_k_m.gguf" \
    "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# 5 - Qwen3-8B
download "qwen3-8b-q4_k_m.gguf" \
    "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/qwen3-8b-q4_k_m.gguf"

# 6 - Llama-3.1-8B
download "llama-3.1-8b-instruct-q4_k_m.gguf" \
    "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# 7 - Gemma-2-9B
download "gemma-2-9b-it-q4_k_m.gguf" \
    "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf"

echo ""
echo "========================================"
echo " All downloads complete!"
echo "========================================"
ls -lh "$MODELS_DIR"
