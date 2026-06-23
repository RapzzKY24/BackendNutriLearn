# Upgrade ke Qwen3-1.7B-Instruct

## 1. Update `requirements.txt`

Ganti:
```
transformers==4.48.3
```
Jadi:
```
transformers>=4.51.0
```

## 2. Update `app/services/llm_service.py`

Ganti:
```python
MODEL_PATHS = {
    "qwen25": "Qwen/Qwen2.5-0.5B-Instruct",
}
```
Jadi:
```python
MODEL_PATHS = {
    "qwen3": "Qwen/Qwen3-1.7B-Instruct",
}
```

## 3. Install ulang dependencies

```bash
cd /home/hyunbinrapz/Documents/projects/ArtificialIntelegence/BackendNutriLearn
source venv/bin/activate
pip install -r requirements.txt
```

## 4. Restart server

Server `--reload` akan otomatis restart. Request pertama akan download model ~3.5GB (cache lokal, sekali aja).
