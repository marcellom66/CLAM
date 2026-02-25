from datetime import datetime, timezone
from clam.config import CONFIG
from clam.memory.short_term import ShortTermBuffer
from clam.memory.long_term import LongTermMemory
from clam.core.models import VectorDBNode

class GarbageCollector:
    """
    Fase 2.C: Gestisce l'evoluzione cognitiva della RAM e la protezione del limite vettoriale:
    Promozione tramite Zettelkasten Linking ed Oblio per via neuronale/temporale.
    """
    def __init__(self, stm: ShortTermBuffer, ltm: LongTermMemory):
        self.stm = stm
        self.ltm = ltm
        self.threshold = CONFIG.memory.short_term.promotion_threshold
        self.decay_minutes = CONFIG.memory.short_term.decay_time_minutes
        self.min_score = CONFIG.memory.short_term.min_score

    async def cycle(self):
        """Task periodico che determina la promozione nei vettori Chroma o la morte del concetto."""
        nodes = await self.stm.get_all_nodes()
        now = datetime.now(timezone.utc)
        
        for node in nodes:
            try:
                # Python 3.11+: fromisoformat accetta la 'Z' finale dell'utc. Puliamo e parsiamo.
                clean_str = node.timestamp_ultimo_accesso.replace("Z", "+00:00")
                last_access = datetime.fromisoformat(clean_str)
                if last_access.tzinfo is None:
                    last_access = last_access.replace(tzinfo=timezone.utc)
            except ValueError:
                # Fallback estremo se stringa rotto
                last_access = now

            age_minutes = (now - last_access).total_seconds() / 60.0
            
            # 1. PROMOZIONE (ZETTELKASTEN GENERATION)
            if node.confidence_score >= self.threshold:
                print(f"[Garbage Collector] Promozione del nodo {node.id_concetto} in Long-Term Memory (Score: {node.confidence_score}). Ricerca legami Zettelkasten...")
                # Esecuzione query semantica di base per scovare parenti nel grafo prima della scrittura
                search_res = await self.ltm.search_semantic(query=node.descrizione, n_results=2)
                
                linked_ids = []
                if search_res and isinstance(search_res, dict) and 'ids' in search_res:
                    if len(search_res['ids']) > 0 and len(search_res['ids'][0]) > 0:
                        linked_ids = search_res['ids'][0] 
                
                meta = {
                    "original_score": node.confidence_score,
                    "contesto_origine": node.contesto_origine,
                    "z_links": ",".join(linked_ids)
                }
                
                lt_node = VectorDBNode(
                    id_concetto=node.id_concetto,
                    descrizione=node.descrizione,
                    metadata=meta
                )
                
                # Consolidiamo come un Fatto Strutturale (Potrebbe in futuro finire nell'Episodica)
                await self.ltm.add_semantic_node(lt_node)
                
                # Spazziamo lo scratchpad volatile
                print(f"[Garbage Collector] Nodo promosso. Eliminazione da Short-Term Buffer.")
                await self.stm.delete_node(node.id_concetto)
                continue
            
            # 2. DECADIMENTO (OBLIO)
            # Vulnerabilità voluta per defaticare il sistema da allucinazioni
            if age_minutes > self.decay_minutes and node.confidence_score < self.min_score:
                print(f"[Garbage Collector] Decadimento. Oblio per il nodo {node.id_concetto} (Age: {age_minutes:.1f}m, Score: {node.confidence_score}).")
                await self.stm.delete_node(node.id_concetto)
