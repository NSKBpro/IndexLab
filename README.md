# VectorDash (local)

## Run locally (no Docker)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
. .venv/bin/activate
pip install --upgrade pip
pip install -e .
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000

## Run with Docker
```bash
docker compose up -d --build
```

## Notes
- First run downloads the embedding model (internet required once).
- Data persists in `data/` (or volume `/data` in Docker).
