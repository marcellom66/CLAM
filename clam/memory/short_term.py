import asyncio
import aiosqlite
from typing import List, Optional
from clam.core.models import MemoryNode

class ShortTermBuffer:
    """
    Gestisce la Memoria Volatile (Scratchpad) usando aiosqlite in memory.
    Implementa rigorosamente asyncio.Lock() per prevenire race conditions letali
    sui database asincroni (Direttiva '10-Year Rule' sulla sicurezza concorrenziale).
    """
    def __init__(self):
        self._db: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def connect(self):
        """Inizializza il database in RAM e crea la tabella se non esiste."""
        self._db = await aiosqlite.connect(":memory:")
        await self._db.execute('''
            CREATE TABLE IF NOT EXISTS memory_nodes (
                id_concetto TEXT PRIMARY KEY,
                descrizione TEXT,
                confidence_score INTEGER,
                timestamp_creazione TEXT,
                timestamp_ultimo_accesso TEXT,
                contesto_origine TEXT
            )
        ''')
        await self._db.commit()

    async def disconnect(self):
        """Chiude e distrugge la connessione."""
        if self._db:
            await self._db.close()

    async def add_node(self, node: MemoryNode):
        """Aggiunge in modo sicuro un nuovo nodo al buffer intercettando eventuali collisioni di ID."""
        async with self._lock:
            if not self._db:
                raise RuntimeError("Errore: Impossibile aggiungere nodo. Database non connesso.")
            
            await self._db.execute('''
                INSERT INTO memory_nodes 
                (id_concetto, descrizione, confidence_score, timestamp_creazione, timestamp_ultimo_accesso, contesto_origine)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                node.id_concetto, 
                node.descrizione, 
                node.confidence_score, 
                node.timestamp_creazione, 
                node.timestamp_ultimo_accesso, 
                node.contesto_origine
            ))
            await self._db.commit()

    async def get_all_nodes(self) -> List[MemoryNode]:
        """Tira fuori tutti i nodi dal buffer (utile per GC e Critic da ciclare)."""
        async with self._lock:
            if not self._db:
                raise RuntimeError("Errore: Impossibile leggere nodi. Database non connesso.")
            
            async with self._db.execute('SELECT * FROM memory_nodes') as cursor:
                rows = await cursor.fetchall()
                
            nodes = []
            for row in rows:
                nodes.append(MemoryNode(
                    id_concetto=row[0],
                    descrizione=row[1],
                    confidence_score=row[2],
                    timestamp_creazione=row[3],
                    timestamp_ultimo_accesso=row[4],
                    contesto_origine=row[5]
                ))
            return nodes

    async def update_score(self, id_concetto: str, delta: int, new_timestamp: str):
        """Aggiorna lo score (positivo o negativo) di un nodo. Invocato tipicamente dal Critic o in rinforzo."""
        async with self._lock:
            if not self._db:
                raise RuntimeError("Errore: Database non connesso.")
            await self._db.execute('''
                UPDATE memory_nodes 
                SET confidence_score = confidence_score + ?, timestamp_ultimo_accesso = ?
                WHERE id_concetto = ?
            ''', (delta, new_timestamp, id_concetto))
            await self._db.commit()

    async def delete_node(self, id_concetto: str):
        """Oblio: Rimuove fisicamente il nodo dal buffer. Usato durante la promozione o il decadimento."""
        async with self._lock:
            if not self._db:
                raise RuntimeError("Errore: Database non connesso.")
            await self._db.execute('DELETE FROM memory_nodes WHERE id_concetto = ?', (id_concetto,))
            await self._db.commit()

    async def clear_all(self):
        """Svuota completamente il buffer in RAM (Formattazione)."""
        async with self._lock:
            if not self._db:
                raise RuntimeError("Errore: Database non connesso.")
            await self._db.execute('DELETE FROM memory_nodes')
            await self._db.commit()
