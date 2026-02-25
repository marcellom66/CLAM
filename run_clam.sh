#!/bin/bash
# Avvia l'ambiente virtuale e il Server API CLAM

echo "[CLAM] Inizializzazione Ambiente Virtuale..."
source .venv/bin/activate || { echo "Runta 'python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt' prima."; exit 1; }

echo "[CLAM] Avvio Motori Cognitivi su FastAPI..."
uvicorn clam.api.server:app --host 127.0.0.1 --port 8000 --reload --reload-exclude "memory_db/*" --reload-exclude "*.db" --reload-exclude "*.sqlite3"
