import os
import aiosqlite
import asyncio
from typing import List, Optional
from clam.core.models import LogicalTriple
from clam.config import CONFIG

class GraphDB:
    """
    Knowledge Graph (Triple Store) implementato su SQLite.
    Garantisce stabilità e previene The 10-Year Rule decay (niente container Neo4j).
    Salva fatti incontestabili: Soggetto -> Relazione -> Oggetto.
    """
    def __init__(self):
        # Genera il path assoluto basato sul file di configurazione
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # punta a 'clam'
        project_dir = os.path.dirname(base_dir) # punta alla root CLAM
        
        # Sostituisce './data' con il path assoluto 'root/data'
        rel_path = CONFIG.memory.long_term.graph_db
        if rel_path.startswith("./"):
            rel_path = rel_path[2:]
            
        self.db_path = os.path.join(project_dir, rel_path)
        
        # Assicuriamoci che la directory 'data' esista
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self._db: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def connect(self):
        """Inizializza il database SQLite su disco e crea la tabella delle triple."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute('''
            CREATE TABLE IF NOT EXISTS triples (
                id_tripla TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_ TEXT NOT NULL,
                confidence INTEGER DEFAULT 1,
                timestamp TEXT NOT NULL
            )
        ''')
        # Creiamo un indice per velocizzare le ricerche incrociate sulle identità
        await self._db.execute('CREATE INDEX IF NOT EXISTS idx_subject ON triples(subject)')
        await self._db.execute('CREATE INDEX IF NOT EXISTS idx_object ON triples(object_)')
        await self._db.commit()

    async def disconnect(self):
        """Chiude la connessione."""
        if self._db:
            await self._db.close()

    async def add_triple(self, triple: LogicalTriple):
        """Inserisce una nuova asserzione logica nel grafo."""
        async with self._lock:
            if not self._db:
                raise RuntimeError("Errore: GraphDB non connesso.")
            
            # Upsert basico per evitare conflitti o sovrascritture di ID
            await self._db.execute('''
                INSERT OR REPLACE INTO triples 
                (id_tripla, subject, predicate, object_, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                triple.id_tripla, 
                triple.subject, 
                triple.predicate, 
                triple.object_, 
                triple.confidence, 
                triple.timestamp
            ))
            await self._db.commit()

    async def get_all_triples(self) -> List[LogicalTriple]:
        """Restituisce tutto il grafo (utile per esplorare o graficare la GUI)."""
        async with self._lock:
            if not self._db:
                raise RuntimeError("Errore: GraphDB non connesso.")
            
            async with self._db.execute('SELECT * FROM triples ORDER BY timestamp DESC') as cursor:
                rows = await cursor.fetchall()
                
            return [
                LogicalTriple(
                    id_tripla=r[0],
                    subject=r[1],
                    predicate=r[2],
                    object_=r[3],
                    confidence=r[4],
                    timestamp=r[5]
                ) for r in rows
            ]

    async def get_triples_by_entity(self, entity_name: str) -> List[LogicalTriple]:
        """
        Retrieval (RAG Logico): Dato un SOGGETTO o un OGGETTO, trova tutti i legami.
        Es: Se cerco 'Utente', torna [Utente, ha_nome, Marcello].
        """
        async with self._lock:
            if not self._db:
                raise RuntimeError("Errore: GraphDB non connesso.")
            
            # Cerca ovunque il termine sia coinvolto (sia come attore che come oggetto)
            query = '''
                SELECT * FROM triples 
                WHERE subject LIKE ? OR object_ LIKE ?
                ORDER BY confidence DESC
            '''
            term = f"%{entity_name}%"
            async with self._db.execute(query, (term, term)) as cursor:
                rows = await cursor.fetchall()
                
            return [
                LogicalTriple(
                    id_tripla=r[0], subject=r[1], predicate=r[2], object_=r[3], 
                    confidence=r[4], timestamp=r[5]
                ) for r in rows
            ]

    async def delete_triple(self, id_tripla: str):
        """Rimuove chirurgicamente una singola verità assoluta."""
        async with self._lock:
            if not self._db:
                raise RuntimeError("Errore: GraphDB non connesso.")
            await self._db.execute('DELETE FROM triples WHERE id_tripla = ?', (id_tripla,))
            await self._db.commit()

    async def delete_triples_by_pattern(self, subject: str, predicate: str, object_: str):
        """Elimina una o più triple che matchano la descrizione esatta (usato dall'LLM per auto-correggersi)."""
        async with self._lock:
            if not self._db:
                raise RuntimeError("Errore: GraphDB non connesso.")
            await self._db.execute('''
                DELETE FROM triples 
                WHERE subject = ? COLLATE NOCASE 
                  AND predicate = ? COLLATE NOCASE 
                  AND object_ = ? COLLATE NOCASE
            ''', (subject, predicate, object_))
            await self._db.commit()

    async def clear_all(self):
        """Formattazione totale per il Reset Memoria."""
        async with self._lock:
            if not self._db:
                raise RuntimeError("Errore: GraphDB non connesso.")
            await self._db.execute('DELETE FROM triples')
            await self._db.commit()
