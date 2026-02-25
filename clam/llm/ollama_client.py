import asyncio
from typing import Dict, Any, Optional, List
import ollama
from clam.config import CONFIG

class OllamaClient:
    """
    Client asincrono per l'interazione con il modello locale (Qwen).
    Abbattendo la latenza di rete si sblocca la possibilità di far girare logiche
    continue in background (Inference, Critic, GC).
    """
    def __init__(self):
        self.model = CONFIG.llm.model
        self.client = ollama.AsyncClient(host=CONFIG.llm.base_url)

    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: str = "", 
        json_format: bool = False,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Genera una risposta testuale (o in JSON strutturato) in modalità asincrona.
        
        chat_history: Lista opzionale di messaggi precedenti [{role: "user"/"assistant", content: "..."}].
                      Vengono inseriti TRA il system prompt e il messaggio corrente, così il modello
                      ha il contesto della conversazione in corso (sa come ti chiami, cosa avete detto, ecc.).
        
        10-Year Rule: In caso di disconnessione o down del container Ollama, blocca ma non fa esplodere l'app.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Inserisce la cronologia della conversazione prima del messaggio corrente.
        # Questo permette al modello di "ricordare" cosa è stato detto nella sessione.
        if chat_history:
            messages.extend(chat_history)
        
        messages.append({"role": "user", "content": prompt})

        print(f"[OllamaClient] Richiesta a '{self.model}' (lunghezza prompt: {len(prompt)}, history: {len(chat_history or [])} msg)")
        try:
            options = {"temperature": CONFIG.llm.temperature}
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                format='json' if json_format else '',
                options=options
            )
            content = response['message']['content']
            print(f"[OllamaClient] Risposta ricevuta (lunghezza: {len(content)})")
            return content
        except Exception as e:
            # Protezione hw/rete, specialmente in locale (laptop scarico o ollama stoppato).
            print(f"[Ollama Error] Fallimento connessione al modello: {e}")
            return ""

