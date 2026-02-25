import uuid
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field

class MemoryNode(BaseModel):
    """
    Rappresenta un singolo concetto o memoria nello Short-Term Buffer.
    Seguendo la '10-Year Rule', usiamo Type Hinting rigoroso ed evitiamo dizionari raw.
    """
    id_concetto: str = Field(default_factory=lambda: str(uuid.uuid4()), description="ID univoco della memoria (UUID in formato stringa per SQLite veloce)")
    descrizione: str = Field(..., description="Il contenuto del concetto/memoria dedotto dall'Inference Engine")
    confidence_score: int = Field(default=1, description="Punteggio di fiducia. += 1 se resiste al Critic, -= 1 se fallisce o se decade")
    timestamp_creazione: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Inizio vita del concetto (ISO 8601 utc)")
    timestamp_ultimo_accesso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Ultimo accesso o rinforzo (ISO 8601 utc)")
    contesto_origine: str = Field(..., description="Dove e come l'agente ha dedotto questa informazione (es. 'User Prompt', 'Internal Debate')")

class VectorDBNode(BaseModel):
    """
    Rappresenta un nodo salvato permanentemente nella Long-Term Memory (ChromaDB).
    I vettori (embeddings) saranno gestiti direttamente da ChromaDB.
    """
    id_concetto: str
    descrizione: str
    metadata: dict = Field(default_factory=dict, description="Metadati strutturati (timestamp, legami Zettelkasten z-links, score originario).")

class LogicalTriple(BaseModel):
    """
    Rappresentazione di un'asserzione logica incrollabile in formato Soggetto-Predicato-Oggetto.
    Base matematica del GraphDB per sconfiggere le allucinazioni sull'identità e sui fatti.
    """
    id_tripla: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: str = Field(..., description="Il soggetto dell'azione, es. 'Utente' o 'CLAM'")
    predicate: str = Field(..., description="L'azione o relazione, es. 'ha_nome'")
    object_: str = Field(..., description="L'oggetto dell'azione, es. 'Marcello'")
    confidence: int = Field(default=1, description="Forza della verità (1-5)")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
