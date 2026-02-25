import asyncio
from datetime import datetime, timezone
from clam.llm.ollama_client import OllamaClient
from clam.memory.short_term import ShortTermBuffer

CRITIC_SYSTEM_PROMPT = """
Sei il Critic di un'architettura cognitiva (rif. Reflexion Paper). Il tuo ruolo è di Avvocato del Diavolo in un dibattito interno.
Riceverai l'astrazione di un concetto dedotto presente nella memoria volatile.
Il tuo compito è esclusivamente cercare contraddizioni, falle logiche o errori tecnici manifesti, rispetto alla tua conoscenza pregressa ingegneristica.
Se la deduzione tecnica è impeccabile e non trovi difetti razionali, rispondi con un'unica parola: "APPROVATO".
Se trovi il minimo difetto, descrivi testualmente all'agente perché la sua assunzione è debole/falsa (Internal Debate Log).
Sii spietatamente logico. Non devi compiacere nessuno.
"""

def _is_user_chatting() -> bool:
    """
    Controlla se c'è una richiesta utente in corso.
    Importiamo il flag QUI (lazy import) per evitare import circolari con server.py.
    
    10-Year Rule: Ollama processa le richieste una alla volta (FIFO).
    Se il Critic accoda una richiesta mentre l'utente sta chattando,
    la risposta dell'utente finisce in coda e CLAM sembra "morto".
    """
    try:
        from clam.api import server
        return server._user_request_active
    except Exception:
        return False

class CriticEngine:
    """
    Fase 2.B: Il motore ad alto scetticismo per avvalorare o abbattere i MemoryNodes.
    """
    def __init__(self, llm_client: OllamaClient, stm: ShortTermBuffer):
        self.llm = llm_client
        self.stm = stm

    async def evaluate_node(self, node_id: str, description: str):
        print(f"[Critic Engine] Analisi del nodo volante: '{description[:50]}...'")
        prompt = f"Scansionamento concetto:\n'{description}'\n\nDetermina se approvarlo o contraddirlo."
        response = await self.llm.generate_response(prompt=prompt, system_prompt=CRITIC_SYSTEM_PROMPT)
        
        now_str = datetime.now(timezone.utc).isoformat()
        if "APPROVATO" in response.upper():
            print(f"[Critic Engine] Esito Positivo! Score +1 per il nodo.")
            await self.stm.update_score(node_id, delta=1, new_timestamp=now_str)
        else:
            print(f"[Critic Engine] Esito Negativo. Score -1. Dibattito:\n{response}")
            await self.stm.update_score(node_id, delta=-1, new_timestamp=now_str)

    async def run_scan(self):
        """Metodo asincrono invocabile per ciclare l'intero buffer a basso priority processing."""
        nodes = await self.stm.get_all_nodes()
        if nodes:
            print(f"[Critic Engine] Avviata scansione periodica su {len(nodes)} nodi in memoria volatile...")
        for node in nodes:
            # CHECKPOINT: se l'utente sta chattando, interrompo subito per liberare Ollama
            if _is_user_chatting():
                print("[Critic Engine] ⏸ Scansione interrotta — utente in chat, priorità alla risposta.")
                return
            await self.evaluate_node(node.id_concetto, node.descrizione)

