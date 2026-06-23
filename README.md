# NutriAI Backend

Backend RAG untuk NutriAI — Gizi & Kesehatan berdasarkan Permenkes.

## Tech Stack

- **FastAPI** — Web framework
- **ChromaDB** — Vector database
- **Qwen 3** — LLM via Hugging Face
- **Sentence Transformers** — Embedding model

## Setup & Running

```bash
# 1. Clone repositori
git clone git@github.com:RapzzKY24/BackendNutriLearn.git
cd BackendNutriLearn

# 2. Buat virtual environment
python3.12 -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Buat file .env (isi token dari admin)
cp .env.example .env

# 5. Ingest dokumen ke ChromaDB
python ingest.py

# 6. Jalankan server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 7. Buka browser: http://localhost:8000/docs
```

## File .env

Buat file `.env` di root project dengan isi:

```
HF_TOKEN=hf_xxx...               # Token Hugging Face
MODEL_NAME=qwen3                  # Nama model
CHROMA_PATH=./chroma_db           # Path ChromaDB
PDF_PATH=./documents/Permenkes Nomor 41 Tahun 2014.pdf
TOP_K=5                           # Jumlah chunk relevan
TEMPERATURE=0.3                   # Kreativitas model
MAX_NEW_TOKENS=512                # Maks token jawaban
```

## API Endpoints

| Method | Path | Deskripsi |
|--------|------|-----------|
| GET | `/` | Health check |
| POST | `/chat` | Tanya NutriAI (RAG) |
| POST | `/bmi` | Hitung BMI |
| POST | `/evaluate` | Evaluasi LLM tanpa konteks |
| POST | `/evaluate/rag` | Evaluasi RAG dengan konteks |
| GET | `/docs` | Swagger UI |

## Catatan

- `.env` dan `chroma_db/` sudah di `.gitignore`, tidak ikut terpush.
- Setiap clone baru, jalankan `python ingest.py` dulu untuk mengisi ChromaDB.
