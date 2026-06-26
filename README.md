# NutriAI Backend — RAG System for Indonesian Nutrition Guidelines

Backend **Retrieval-Augmented Generation (RAG)** untuk NutriAI. Sistem ini menjawab pertanyaan tentang gizi dan kesehatan berdasarkan **Peraturan Menteri Kesehatan (Permenkes) No. 41 Tahun 2014 tentang Pedoman Gizi Seimbang**.

Dibangun dengan **FastAPI**, menggunakan **LLM lokal (format GGUF)** dan **ChromaDB** sebagai vector database.

---

## Architecture Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                        CLIENT (Frontend / API)                   │
└─────────────────────────┬────────────────────────────────────────┘
                          │ POST /chat { question: "..." }
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  app/api/chat.py ← menerima request, validasi input             │
│  └─→ app/services/rag_service.py ← orchestrator RAG             │
│       │                                                         │
│       ├─→ app/services/retriever_service.py                     │
│       │     └─→ embedding_service (SentenceTransformer)         │
│       │     └─→ ChromaDB vector search                          │
│       │         ↑ relevan chunks dari dokumen                   │
│       │                                                         │
│       ├─→ context + question → format prompt                    │
│       │                                                         │
│       └─→ app/services/llm_service.py                           │
│             └─→ llama-cpp-python (GGUF model)                   │
│                 ↑ jawaban + sumber halaman                       │
│                                                                  │
│  app/api/bmi.py ← hitung BMI (tidak pakai LLM)                  │
│  app/api/evaluation.py ← evaluasi performa model & RAG          │
│  app/api/health.py ← cek status server & model                  │
└──────────────────────────────────────────────────────────────────┘
```

### Alur RAG — Step by Step

1. User mengirim pertanyaan ke `POST /chat`
2. `RAGService` memanggil `RetrieverService` untuk mencari konteks relevan
3. `RetrieverService` meng-*embed* pertanyaan pakai **SentenceTransformer** (`all-MiniLM-L6-v2`)
4. ChromaDB mencari **5 chunk dokumen** terdekat secara semantik (cosine similarity)
5. Chunk yang relevan diformat jadi prompt bersama pertanyaan user
6. Prompt dikirim ke **LLM lokal (Qwen3 GGUF)** via `llama-cpp-python`
7. LLM menjawab berdasarkan konteks, mencantumkan nomor halaman sumber
8. Jawaban + sumber dikembalikan ke user

---

## Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Web Framework | **FastAPI** (Python) |
| LLM | **Qwen3** (GGUF format) via `llama-cpp-python` |
| Embedding Model | **sentence-transformers/all-MiniLM-L6-v2** (384d) |
| Vector Database | **ChromaDB** (persistent, cosine similarity) |
| PDF Parser | **pypdf** |
| HTTP Server | **Uvicorn** |

---

## Project Structure

```
BackendNutriLearn/
├── .env                          # Konfigurasi environment (TIDAK DI-PUSH)
├── .gitignore
├── requirements.txt              # Dependencies Python
├── ingest.py                     # Script: PDF → ChromaDB (jalankan sekali)
├── README.md                     # File ini
│
├── app/
│   ├── main.py                   # Entry point FastAPI, register routers
│   │
│   ├── core/
│   │   ├── config.py             # Settings dari .env (Pydantic)
│   │   ├── logger.py             # Logging setup
│   │   └── prompt.py             # Prompt template untuk LLM
│   │
│   ├── api/
│   │   ├── chat.py               # POST /chat — RAG Q&A
│   │   ├── bmi.py                # POST /bmi — Kalkulator BMI
│   │   ├── evaluation.py         # POST /evaluate & /evaluate/rag
│   │   └── health.py             # GET /health — Status server
│   │
│   ├── services/
│   │   ├── rag_service.py        # Orchestrator RAG (retrieve → prompt → generate)
│   │   ├── llm_service.py        # Load & inference GGUF model (llama-cpp-python)
│   │   ├── embedding_service.py  # Load & inference embedding model (SentenceTransformer)
│   │   ├── retriever_service.py  # ChromaDB client + query vector search
│   │   ├── ingestion_service.py  # PDF → chunking → embedding → ChromaDB
│   │   └── eval_rag_service.py   # Evaluasi metrics RAG (relevance, faithfulness)
│   │
│   └── models/
│       ├── request.py            # Pydantic schemas untuk request body
│       └── response.py           # Pydantic schemas untuk response
│
├── models/                       # Folder GGUF model files (TIDAK DI-PUSH)
│   ├── qwen3-1.7b-q4_k_m.gguf
│   └── qwen3-4b-q4_k_m.gguf
│
├── chroma_db/                    # Persistent ChromaDB storage (TIDAK DI-PUSH)
│
└── documents/
    └── Permenkes Nomor 41 Tahun 2014.pdf  # Dokumen referensi
```

---

## Setup & Installation

> **Catatan untuk MacBook (Apple Silicon M1/M2/M3/M4):**  
> `llama-cpp-python` sudah support **Metal GPU acceleration** secara otomatis. Tidak perlu konfigurasi tambahan.

### 1. Clone repositori

```bash
git clone git@github.com:RapzzKY24/BackendNutriLearn.git
cd BackendNutriLearn
```

### 2. Buat & aktifkan virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Jika mau maksimalin GPU MacBook (Metal), install llama-cpp-python dengan flag Metal:
> ```bash
> GGML_METAL=1 pip install llama-cpp-python --force-reinstall --no-cache-dir
> ```
> Tapi biasanya `pip install -r requirements.txt` sudah cukup.

### 4. Download model GGUF

Unduh model **Qwen3** dalam format GGUF. Letakkan di folder `models/`.

**Opsi A — Qwen3-4B (direkomendasikan, balance speed & quality):**

```bash
# Download dari HuggingFace
# Cari file GGUF Qwen3-4B di https://huggingface.co/Qwen
# Contoh: qwen3-4b-q4_k_m.gguf
```

**Opsi B — Qwen3-1.7B (lebih ringan, cepat di CPU):**

```bash
# Download dari HuggingFace
# Contoh: qwen3-1.7b-q4_k_m.gguf
```

> **Tips download:**  
> - Cari file GGUF di HuggingFace model page (filter: file type `.gguf`)  
> - Atau tambahkan via `git lfs`
> - Simpan file `.gguf` ke folder `models/` di root project

### 5. Konfigurasi .env

Buat file `.env` di root project:

```env
GGUF_MODEL_PATH=./models/qwen3-4b-q4_k_m.gguf
CHROMA_PATH=./chroma_db
PDF_PATH=./documents/Permenkes Nomor 41 Tahun 2014.pdf
TOP_K=5
TEMPERATURE=0.3
MAX_NEW_TOKENS=512
```

### 6. Ingest dokumen ke ChromaDB (wajib sekali)

```bash
python ingest.py
```

Proses ini membaca PDF, memecah jadi chunk, membuat embedding, dan menyimpan ke ChromaDB.  
**Harus dijalankan setiap kali PDF diganti atau ChromaDB dihapus.**

### 7. Jalankan server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Buka browser: [http://localhost:8000/docs](http://localhost:8000/docs) — Swagger UI akan muncul.

---

## Configuration (.env)

| Variable | Default | Deskripsi |
|----------|---------|-----------|
| `GGUF_MODEL_PATH` | `./models/qwen3-4b-q4_k_m.gguf` | Path ke file model GGUF |
| `CHROMA_PATH` | `./chroma_db` | Path folder ChromaDB persistent storage |
| `PDF_PATH` | `./documents/Permenkes Nomor 41 Tahun 2014.pdf` | Path ke dokumen PDF referensi |
| `TOP_K` | `5` | Jumlah chunk relevan yang diambil dari ChromaDB |
| `TEMPERATURE` | `0.3` | Creative control LLM (0.0 = deterministik, 1.0 = kreatif) |
| `MAX_NEW_TOKENS` | `512` | Maksimal token untuk jawaban LLM |

---

## API Endpoints

### `GET /`

Root endpoint — cek apakah server berjalan.

**Response:**
```json
{
  "message": "NutriAI RAG Service is running",
  "docs": "/docs"
}
```

---

### `GET /health`

Cek status server dan model.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### `POST /chat`

**Endpoint utama RAG.** Kirim pertanyaan, dapat jawaban dengan sumber halaman.

**Request:**
```json
{
  "question": "Apa itu gizi seimbang?"
}
```

**Response:**
```json
{
  "answer": "Gizi seimbang adalah susunan pangan sehari-hari... (Halaman 15)\n\n---\n📖 **Sumber:** Halaman 15, Halaman 16",
  "sources": [
    { "page": 15 },
    { "page": 16 }
  ]
}
```

**Flow:** `question → retrieve dari ChromaDB → format prompt → LLM generate → jawaban + sources`

---

### `POST /bmi`

Hitung BMI (tidak pakai LLM).

**Request:**
```json
{
  "weight": 65,
  "height": 170
}
```

**Response:**
```json
{
  "bmi": 22.5,
  "category": "Normal"
}
```

---

### `POST /evaluate`

Evaluasi LLM **tanpa konteks RAG**. Cek kemampuan model menjawab dari pengetahuan internalnya saja.

**Request:**
```json
{
  "question": "Apa itu gizi seimbang?"
}
```

**Response:**
```json
{
  "answer": "Gizi seimbang adalah...",
  "latency": 3.45
}
```

`latency` dalam detik — berguna untuk benchmarking kecepatan antar model.

---

### `POST /evaluate/rag`

Evaluasi **RAG pipeline secara kuantitatif**. Menghasilkan 3 metrics untuk mengukur kualitas sistem.

**Request:**
```json
{
  "question": "Apa itu gizi seimbang?",
  "answer": ""  // Kosongkan untuk auto-generate dari RAG, atau isi sendiri
}
```

**Response:**
```json
{
  "question": "Apa itu gizi seimbang?",
  "answer": "Gizi seimbang adalah...",
  "context_sources": [{"page": 15}, {"page": 16}],
  "metrics": {
    "context_relevance": 0.85,
    "answer_faithfulness": 0.92,
    "answer_relevance": 0.88,
    "total_claims": 12,
    "faithful_claims": 11,
    "relevant_sentences": 8,
    "total_sentences": 10,
    "analysis": "Konteks yang diberikan sangat relevan dengan pertanyaan. Jawaban sangat setia pada konteks yang diberikan. Jawaban sangat relevan dengan pertanyaan."
  }
}
```

#### Penjelasan Metrics:

| Metric | Range | Arti |
|--------|-------|------|
| `context_relevance` | 0.0 – 1.0 | Seberapa relevan dokumen yang di-retrieve dengan pertanyaan |
| `answer_faithfulness` | 0.0 – 1.0 | Seberapa setia jawaban pada konteks (tidak halusinasi) |
| `answer_relevance` | 0.0 – 1.0 | Seberapa relevan jawaban dengan pertanyaan |
| `total_claims` | int | Jumlah klaim/sentence dalam jawaban |
| `faithful_claims` | int | Jumlah klaim yang didukung konteks |
| `analysis` | text | Interpretasi hasil dalam bahasa Indonesia |

---

## Cara Ganti Model / Testing Model Lain

Fitur utama project ini adalah **mudah berganti model LLM**. Begini caranya:

### Step 1: Download Model GGUF Baru

Cari model GGUF dari sumber seperti:
- **HuggingFace** — cari `Qwen/Qwen3-4B-GGUF` atau model GGUF lain
- **Model lain yang kompatibel:** llama-cpp-python support model apa pun di format GGUF (Llama 3, Mistral, Gemma, dll.)

```bash
# Contoh download model lain ke folder models/
# Letakkan file .gguf di ./models/
```

### Step 2: Update .env

Ubah `GGUF_MODEL_PATH` di `.env`:

```env
# Contoh ganti ke model yang sudah ada
GGUF_MODEL_PATH=./models/qwen3-4b-q4_k_m.gguf

# Atau ganti ke model lain
# GGUF_MODEL_PATH=./models/model-baru-saya.gguf
```

### Step 3: Restart Server

Matikan server (Ctrl+C) lalu jalankan ulang:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Model akan otomatis di-load saat request pertama masuk (**lazy loading**).

### Step 4: Test

- `GET /health` — cek `"model_loaded": true`
- `POST /chat` — test jawaban dengan pertanyaan
- `POST /evaluate` — test kecepatan (latency)
- `POST /evaluate/rag` — test kualitas RAG metrics

### Tips Memilih Model

| Model | Ukuran | Kecepatan (Mac) | Kualitas |
|-------|--------|-----------------|----------|
| Qwen3-1.7B Q4 | ~1 GB | Sangat Cepat | Cukup |
| Qwen3-4B Q4 | ~2.5 GB | Cepat | Baik |
| Qwen3-8B Q4 | ~5 GB | Sedang | Sangat Baik |
| Llama 3.2 3B | ~2 GB | Cepat | Baik |

> **⚠️ Penting untuk MacBook:**  
> - Mac dengan RAM 8GB → maksimal model 4B Q4 (jangan 8B)  
> - Mac dengan RAM 16GB+ → model 8B Q4 masih oke  
> - `llama-cpp-python` otomatis pakai **Metal GPU** di Mac Silicon — jauh lebih cepat dari CPU

---

## Ganti Dokumen Referensi (PDF)

Proyek ini menggunakan 1 PDF sebagai sumber pengetahuan. Jika ingin ganti dokumen:

1. Letakkan file PDF baru di folder `documents/`
2. Update `PDF_PATH` di `.env`:
   ```env
   PDF_PATH=./documents/nama-file-baru.pdf
   ```
3. Jalankan ulang ingestion:
   ```bash
   python ingest.py
   ```
   ChromaDB akan dibuat ulang dengan konten dari PDF baru.

---

## Development Notes

### File yang Tidak Di-Push ke GitHub

```
.env              ← Environment variables (rahasia)
chroma_db/        ← Vector database (bisa di-rebuild)
models/           ← GGUF file (terlalu besar)
venv/             ← Virtual environment
__pycache__/      → Cache Python
.opencode/        → Konfigurasi opencode (developer tools)
```

### Logging

Semua log ada di console dengan format:
```
2026-06-26 10:00:00 | nutriai | INFO | Model loaded in 3.45s
```

Berguna untuk debugging dan monitoring performa.

### Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `Model loaded: false` | Cek path di `.env`, pastikan file `.gguf` ada |
| `ChromaDB belum di-populate` | Jalankan `python ingest.py` |
| `Out of memory` | Pakai model yang lebih kecil (1.7B instead of 4B) |
| Lambat di Mac | Pastikan `llama-cpp-python` terinstall dengan Metal support |
| Answer tidak mencantumkan halaman | Prompt meminta LLM mencantumkan halaman — model kecil kadang skip. Coba model lebih besar |

---

## Quick Reference Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Ingest dokumen
python ingest.py

# Run server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question":"Apa itu gizi seimbang?"}'
```
