import asyncio
import os
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from contextlib import asynccontextmanager

from clam.config import CONFIG
from clam.llm.ollama_client import OllamaClient
from clam.memory.short_term import ShortTermBuffer
from clam.memory.long_term import LongTermMemory
from clam.memory.graph_db import GraphDB
from clam.engines.inference import InferenceEngine
from clam.engines.critic import CriticEngine
from clam.engines.gc import GarbageCollector
from clam.core.agent import ClamAgent
from clam.core.models import LogicalTriple
from clam.core.knowledge_renderer import KnowledgeRenderer
from clam.core.knowledge_schema import normalize_predicate

# Iniziamo lo state globale
stm = ShortTermBuffer()
ltm = LongTermMemory()
gdb = GraphDB()
llm = OllamaClient()

inference_engine = InferenceEngine(llm, stm, ltm, gdb)
critic_engine = CriticEngine(llm, stm)
gc_engine = GarbageCollector(stm, ltm)
agent = ClamAgent(llm, stm, ltm, gdb)
knowledge_renderer = KnowledgeRenderer()

# Path al file seed (relativo alla root del progetto)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SEED_FILE_PATH = os.path.join(_PROJECT_ROOT, "seed_truths.yaml")

class ConnectionManager:
    """Gestisce l'elenco delle connessioni WebSocket attive per fare PUSH statelet al frontend Vis.js"""
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WebSocket] Nuovo client frontend connesso. Totale: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

import sys
class WSTerminal:
    """Intercetta lo standard output per convogliarlo nella console javascript del browser"""
    def __init__(self, manager):
        self.manager = manager
        self.original_stdout = sys.stdout

    def write(self, message):
        self.original_stdout.write(message)
        if message.strip():
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.manager.broadcast({"type": "console_log", "message": message.strip()}))
            except RuntimeError:
                pass

    def flush(self):
        self.original_stdout.flush()

sys.stdout = WSTerminal(manager)
# Flag globale: quando l'utente sta chattando, i motori di background si mettono in pausa.
# Ollama processa le richieste una alla volta (FIFO), quindi se il Critic occupa il modello,
# la risposta dell'utente finisce in coda e CLAM sembra "non rispondere".
_user_request_active = False

# Background Tasks
async def background_loop():
    """Il battito cardiaco asincrono dei motori cognitivi (Fase 2)."""
    while True:
        try:
            await asyncio.sleep(15)  # Background loop: 15s
            
            # 1. Telemetria Visiva (Dashboard) — sempre attiva, non chiama l'LLM
            nodes = await stm.get_all_nodes()
            ltm_data = await ltm.get_recent_semantic(limit=50)
            triples = await gdb.get_all_triples()
            
            await manager.broadcast({
                "type": "telemetry",
                "stm_count": len(nodes),
                "nodes": [n.model_dump() for n in nodes],
                "ltm_count": len(ltm_data.get("ids", [])),
                "ltm_data": ltm_data,
                "graph_count": len(triples),
                "triples": [t.model_dump() for t in triples]
            })
            
            # 2. Motori iterativi — SOLO se l'utente NON sta aspettando una risposta
            if not _user_request_active:
                # NOTA: Critic DISABILITATO — con qwen2.5:3b dice SEMPRE "Esito Negativo"
                # e blocca Ollama ogni 15s impedendo le risposte all'utente.
                # Con il dual-write i fatti vanno direttamente in LTM senza bisogno del Critic.
                # Riabilitare con un modello più grande (es. qwen3:8b).
                # await critic_engine.run_scan()
                await gc_engine.cycle()
            else:
                print("[Core Loop] Motori in pausa — richiesta utente in corso...")
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[Core Loop Error] Impossibile completare il ciclo asincrono: {e}")

async def _load_seed_truths():
    """Carica i fatti fondamentali dal file seed_truths.yaml nel GraphDB."""
    if not os.path.exists(_SEED_FILE_PATH):
        print(f"[Seed] File seed non trovato: {_SEED_FILE_PATH}")
        return 0

    with open(_SEED_FILE_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    truths = data.get("seed_truths", [])
    count = 0
    for t in truths:
        sub = t.get("subject")
        pred = normalize_predicate(t.get("predicate", ""))
        obj = t.get("object")
        if sub and pred and obj:
            triple = LogicalTriple(subject=sub, predicate=pred, object_=obj, confidence=5)
            await gdb.add_triple(triple)
            count += 1
            print(f"[Seed] 🌱 Caricato: {sub} -> {pred} -> {obj}")

    return count


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup FastAPI startup
    await stm.connect()
    await gdb.connect()

    # Auto-seed: se il GraphDB è vuoto, carica i fatti fondamentali
    existing_triples = await gdb.get_all_triples()
    if len(existing_triples) == 0:
        count = await _load_seed_truths()
        print(f"[Seed] ✅ Caricati {count} fatti fondamentali dal seed file")

    loop_task = asyncio.create_task(background_loop())
    yield
    # Shutdown
    loop_task.cancel()
    await stm.disconnect()
    await gdb.disconnect()

# Engine Core Interface
app = FastAPI(title="CLAM OS - Brain Endpoint", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """Fase principale: Il Prompting passa da qui e subisce pre-post processing cognitivo."""
    global _user_request_active
    print(f"\n[API] Ricevuto nuovo prompt utente: {req.message}")
    
    # Blocca i motori di background per dare priorità alla risposta utente
    _user_request_active = True
    try:
        response = await agent.generate_reply(req.message)
        print(f"[API] Risposta generata, accodo l'Inference Engine in background...")
        
        # Fire and Forget: L'Inference deve avvenire in background
        asyncio.create_task(inference_engine.perceive(req.message, response))
    finally:
        # Riattiva i motori di background DOPO che la risposta è stata generata
        _user_request_active = False
    
    return {"reply": response}

@app.delete("/api/memory")
async def clear_memory_endpoint():
    """Cancella tutti i ricordi STM e LTM su richiesta esplicita dell'utente."""
    await stm.clear_all()
    await ltm.clear_all()
    await gdb.clear_all()
    print("\n[API] 🧹 Formattazione Memoria... (STM, LTM & GraphDB Svuotate)\n")
    return {"status": "ok"}

@app.delete("/api/memory/item/{store}/{item_id}")
async def delete_memory_item(store: str, item_id: str):
    """Cancella chirurgicamente una singola memoria."""
    print(f"\n[API] 🗑️ Rimuovo {item_id} da {store}")
    if store == "stm":
        await stm.delete_node(item_id)
    elif store == "ltm":
        await ltm.delete_semantic_node(item_id)
    elif store == "graph":
        await gdb.delete_triple(item_id)
    return {"status": "ok"}

# ─────────────────────────────────────────────────────────────────
# API per il Knowledge Graph strutturato
# ─────────────────────────────────────────────────────────────────

@app.get("/api/knowledge/document")
async def get_knowledge_document():
    """Restituisce il documento di conoscenza strutturato come lo vedrà l'LLM."""
    doc = await knowledge_renderer.render_knowledge_document(gdb)
    return {"document": doc if doc else "Nessun fatto registrato."}

class TripleRequest(BaseModel):
    subject: str
    predicate: str
    object_: str

@app.post("/api/knowledge/triple")
async def add_knowledge_triple(req: TripleRequest):
    """Aggiunge manualmente una tripla al Knowledge Graph con predicato normalizzato."""
    normalized_pred = normalize_predicate(req.predicate)
    triple = LogicalTriple(subject=req.subject, predicate=normalized_pred, object_=req.object_, confidence=5)
    await gdb.add_triple(triple)
    print(f"[API] ➕ Tripla aggiunta: {req.subject} -> {normalized_pred} -> {req.object_}")
    return {"status": "ok", "id": triple.id_tripla, "normalized_predicate": normalized_pred}

@app.delete("/api/knowledge/triple/{triple_id}")
async def delete_knowledge_triple(triple_id: str):
    """Elimina una singola tripla dal Knowledge Graph."""
    await gdb.delete_triple(triple_id)
    print(f"[API] 🗑️ Tripla eliminata: {triple_id}")
    return {"status": "ok"}

@app.post("/api/knowledge/seed")
async def load_seed_truths():
    """Carica (o ri-carica) i fatti fondamentali dal file seed_truths.yaml."""
    count = await _load_seed_truths()
    print(f"[API] 🌱 Seed caricato: {count} fatti")
    return {"status": "ok", "loaded": count}

@app.post("/api/knowledge/reset-and-seed")
async def reset_and_seed():
    """Reset totale + ri-caricamento dei fatti fondamentali. Riparte da zero."""
    await stm.clear_all()
    await ltm.clear_all()
    await gdb.clear_all()
    count = await _load_seed_truths()
    print(f"[API] 🔄 Reset completo + Seed: {count} fatti caricati")
    return {"status": "ok", "loaded": count}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Connessione pura TCP per il render visivo in Javascript senza delay da HTTP Polling."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        manager.disconnect(websocket)

def start_server():
    uvicorn.run("clam.api.server:app", host=CONFIG.api.host, port=CONFIG.api.port, log_level="info")

if __name__ == "__main__":
    start_server()
