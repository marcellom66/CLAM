import asyncio
import chromadb
from clam.core.models import VectorDBNode
from clam.config import CONFIG

class LongTermMemory:
    """
    Gestisce la Memoria Consolidata (Semantica ed Episodica) usando ChromaDB in locale su persistenza disco.
    Separiamo le Collezioni Vettoriali per rispettare il paradigma CoALA (Experiences vs Facts).
    Avvolge i driver sincroni di chromadb in blocchi asincroni per non fermare l'event loop di asyncio.
    """
    def __init__(self):
        self._lock = asyncio.Lock()
        
        # Il path di persistenza evita di fargli scaricare un server intero, creando DB localmente.
        self.client = chromadb.PersistentClient(path=CONFIG.memory.long_term.path)
        
        self.semantic_collection = self.client.get_or_create_collection(name=CONFIG.memory.long_term.semantic_collection)
        self.episodic_collection = self.client.get_or_create_collection(name=CONFIG.memory.long_term.episodic_collection)

    async def add_semantic_node(self, node: VectorDBNode):
        """
        [Fatti] Inserisce informazioni oggettive e preferenze statiche. 
        Costituiscono l'identità operativa dell'agente.
        """
        async with self._lock:
            self.semantic_collection.add(
                documents=[node.descrizione],
                metadatas=[node.metadata],
                ids=[node.id_concetto]
            )

    async def add_episodic_node(self, node: VectorDBNode):
        """
        [Esperienze] Inserisce log e sequenze decisionali. Il database per "non ripetere due volte l'errore".
        """
        async with self._lock:
            self.episodic_collection.add(
                documents=[node.descrizione],
                metadatas=[node.metadata],
                ids=[node.id_concetto]
            )

    async def search_semantic(self, query: str, n_results: int = 3) -> dict:
        """Recupero Vettoriale sui Fatti per la fase di Familiarity Check."""
        async with self._lock:
            return self.semantic_collection.query(
                query_texts=[query],
                n_results=n_results
            )

    async def search_episodic(self, query: str, n_results: int = 3) -> dict:
        """Recupero Vettoriale sulle Esperienze per la fase complessa di Recollection."""
        async with self._lock:
            return self.episodic_collection.query(
                query_texts=[query],
                n_results=n_results
            )

    async def get_recent_semantic(self, limit: int = 50) -> dict:
        """Recupera gli ultimi fatti consolidati nella Memoria Semantica per la Dashboard testuale."""
        async with self._lock:
            return self.semantic_collection.get(limit=limit)

    async def delete_semantic_node(self, id_concetto: str):
        """Elimina chirurgicamente un fatto dalla memoria a lungo termine."""
        async with self._lock:
            self.semantic_collection.delete(ids=[id_concetto])

    async def clear_all(self):
        """Oblio totale: distrugge e ricrea fisicamente le collezioni vettoriali."""
        async with self._lock:
            try:
                self.client.delete_collection(name=CONFIG.memory.long_term.semantic_collection)
                self.client.delete_collection(name=CONFIG.memory.long_term.episodic_collection)
            except ValueError:
                pass # Se la collezione non esiste, chroma alza un value error che possiamo skippare
            
            self.semantic_collection = self.client.get_or_create_collection(name=CONFIG.memory.long_term.semantic_collection)
            self.episodic_collection = self.client.get_or_create_collection(name=CONFIG.memory.long_term.episodic_collection)
