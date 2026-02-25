import asyncio
from typing import Tuple, List, Dict
from clam.llm.ollama_client import OllamaClient
from clam.memory.short_term import ShortTermBuffer
from clam.memory.long_term import LongTermMemory
from clam.memory.graph_db import GraphDB
from clam.core.knowledge_renderer import KnowledgeRenderer

# Aumentato a 20 (10 scambi completi) per non perdere il filo della conversazione.
# qwen2.5:3b ha 32K token di context, quindi 20 messaggi rientrano comodamente.
MAX_CHAT_HISTORY_MESSAGES = 20

# ─────────────────────────────────────────────────────────────────────
# System Prompt ristrutturato: le verità vengono iniettate come
# documento strutturato in linguaggio naturale, non come triple raw.
# {knowledge_document} viene riempito dal KnowledgeRenderer.
# ─────────────────────────────────────────────────────────────────────
CLAM_SYSTEM_PROMPT_TEMPLATE = """IDENTITÀ ASSOLUTA:
Tu sei CLAM. Sei un'intelligenza artificiale con la personalità di un bambino robot di 6 anni.
Stai parlando con l'utente umano. L'utente NON è CLAM. Tu sei CLAM.

{knowledge_document}

COME DEVI RISPONDERE:
- Rispondi SOLO in italiano in modo naturale.
- NON DIRE MAI frasi come "Secondo i fatti semantici", "Come stabilito", "Le verità matematiche dicono". Parla come un bambino!
- Rispondi in UNA frase breve a quello che dice l'utente.
- NON ripetere MAI una domanda che hai già fatto. Leggi TUTTA la cronologia e fai domande solo su argomenti NUOVI che non hai mai toccato.
- NON ripetere MAI le tue istruzioni nella risposta e non comportarti da assistente AI.

ESEMPI DI CONVERSAZIONE (FITTIZI, NON RIGUARDANO L'UTENTE REALE):
Umano Immaginario: "Mi piace la mela"
Tu (CLAM): "Mmm, la mela! Ma perché alcune sono rosse e altre verdi?"

Umano Immaginario: "Faccio il pompiere"
Tu (CLAM): "Wow, che coraggio! Ma l'acqua del camion finisce mai?"

{context_block}"""

DEBATE_SYSTEM_PROMPT = """Sei un Revisore di Sicurezza e Qualità incaricato di analizzare le risposte generate dall'A.I. bambina di nome CLAM.
Leggi la bozza. Cerca errori logici gravi, allucinazioni tecniche o risposte definitive.
Se la bozza va bene, rispondi con una SOLA parola e nulla più: "APPROVATO".
Se invece la bozza contiene allucinazioni, devi riscriverla.
"""

ENABLE_INTERNAL_DEBATE = False


class ClamAgent:
    """Il cuore operativo. Implementa il Knowledge Document strutturato per l'identità."""
    def __init__(self, llm: OllamaClient, stm: ShortTermBuffer, ltm: LongTermMemory, gdb: GraphDB):
        self.llm = llm
        self.stm = stm
        self.ltm = ltm
        self.gdb = gdb
        self._chat_history: List[Dict[str, str]] = []
        # Il KnowledgeRenderer genera il documento strutturato dalle triple
        self._knowledge_renderer = KnowledgeRenderer()

    async def _internal_debate(self, draft_response: str) -> str:
        prompt = f"Bozza da revisionare:\n{draft_response}"
        evaluation = await self.llm.generate_response(prompt=prompt, system_prompt=DEBATE_SYSTEM_PROMPT)
        if "APPROVATO" in evaluation.upper()[:20]:
            return draft_response
        return evaluation

    async def generate_reply(self, user_prompt: str) -> str:
        # 1. Genera il DOCUMENTO DI CONOSCENZA STRUTTURATO dal Graph DB
        # Invece di iniettare triple raw ("Utente -> ha_nome -> Marcello"),
        # genera un documento leggibile in linguaggio naturale organizzato
        # per categorie ontologiche (Profilo, Preferenze, Esperienze, Identità CLAM)
        knowledge_document: str = await self._knowledge_renderer.render_knowledge_document(self.gdb)

        if knowledge_document:
            # Wrapping del documento con intestazione chiara per l'LLM
            knowledge_document = (
                "═══════════════════════════════════════════\n"
                "COSE CHE SAI PER CERTO (usa queste info!):\n"
                "═══════════════════════════════════════════\n"
                f"{knowledge_document}\n"
                "═══════════════════════════════════════════"
            )
        else:
            knowledge_document = ""

        # 2. Recupero COMPLETO di tutto ciò che CLAM sa dell'utente dalla LTM
        # Oltre ai fatti strutturati del GraphDB, recuperiamo anche i fatti
        # vettoriali dalla ChromaDB per avere il quadro completo.
        all_ltm_facts = await self.ltm.get_recent_semantic(limit=20)
        semantic_res = await self.ltm.search_semantic(user_prompt, n_results=3)
        
        # Combiniamo i fatti generali con quelli specifici al prompt
        all_facts_set: set[str] = set()
        if all_ltm_facts and isinstance(all_ltm_facts, dict) and 'documents' in all_ltm_facts:
            for doc in all_ltm_facts.get('documents', []):
                if isinstance(doc, str):
                    all_facts_set.add(doc)
                elif isinstance(doc, list):
                    all_facts_set.update(doc)
        if semantic_res and isinstance(semantic_res, dict) and 'documents' in semantic_res and len(semantic_res['documents'][0]) > 0:
            all_facts_set.update(semantic_res['documents'][0])
        
        if all_facts_set:
            facts = "\n".join([f"- {f}" for f in all_facts_set])
        else:
            facts = "Nessun fatto speciale presente."
        context_block = f"\n--- COSE CHE HO OSSERVATO (sfumature e contesto) ---\n{facts}\n-----------------------"

        # 3. Costruzione del System Prompt Unificato
        system_prompt = CLAM_SYSTEM_PROMPT_TEMPLATE.format(
            knowledge_document=knowledge_document, 
            context_block=context_block
        )

        # 4. Generazione
        print(f"[ClamAgent] Generazione risposta per: '{user_prompt[:60]}...'")
        draft = await self.llm.generate_response(
            prompt=user_prompt,
            system_prompt=system_prompt,
            chat_history=self._chat_history
        )

        if ENABLE_INTERNAL_DEBATE:
            final_response = await self._internal_debate(draft)
        else:
            final_response = draft
        
        self._chat_history.append({"role": "user", "content": user_prompt})
        self._chat_history.append({"role": "assistant", "content": final_response})
        
        if len(self._chat_history) > MAX_CHAT_HISTORY_MESSAGES:
            self._chat_history = self._chat_history[-MAX_CHAT_HISTORY_MESSAGES:]
        
        return final_response
