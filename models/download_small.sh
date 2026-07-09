#!/bin/bash
set -e

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

echo "========================================"
echo " Downloading 4 small benchmark models"
echo " Total: ~6 GB"
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

# 1 - Qwen3-1.7B (existing)
download "qwen3-1.7b-q4_k_m.gguf" \
    "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/qwen3-1.7b-q4_k_m.gguf"

# 2 - Qwen2.5-3B (1.8GB)
download "qwen2.5-3b-instruct-q4_k_m.gguf" \
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

# 3 - Qwen3-4B (2.5GB)
download "qwen3-4b-q4_k_m.gguf" \
    "https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen_Qwen3-4B-Q4_K_M.gguf"

# 4 - Llama-3.2-3B (1.8GB)
download "llama-3.2-3b-instruct-q4_k_m.gguf" \
    "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

echo ""
echo "========================================"
echo " All downloads complete!"
echo "========================================"
ls -lh "$MODELS_DIR"
