import asyncio
from typing import Tuple, List, Dict
from clam.llm.ollama_client import OllamaClient
from clam.memory.short_term import ShortTermBuffer
from clam.memory.long_term import LongTermMemory
from clam.memory.graph_db import GraphDB
from clam.core.knowledge_renderer import KnowledgeRenderer
from clam.core.locales import get_clam_system_prompt, get_debate_prompt, get_knowledge_strings
from clam.config import CONFIG

# Maximum number of messages kept in chat history.
# qwen2.5:3b has a 32K token context, so 20 messages fit comfortably.
MAX_CHAT_HISTORY_MESSAGES = 20

# Language is read once at startup from config.yaml.
# All prompts are built lazily at runtime via locales.get_*().
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
        debate_prompt = get_debate_prompt(CONFIG.language)
        prompt = f"Draft to review:\n{draft_response}"
        evaluation = await self.llm.generate_response(prompt=prompt, system_prompt=debate_prompt)
        if "APPROV" in evaluation.upper()[:20]:  # matches APPROVATO / APPROVED / APROBADO / GENEHMIGT
            return draft_response
        return evaluation

    async def generate_reply(self, user_prompt: str) -> str:
        lang: str = CONFIG.language
        loc = get_knowledge_strings(lang)

        # 1. Build the STRUCTURED KNOWLEDGE DOCUMENT from the Graph DB.
        # Instead of injecting raw triples ("Utente -> ha_nome -> Marcello"),
        # we generate a readable natural-language document organised by
        # ontological categories (Profile, Preferences, Experiences, CLAM Identity).
        knowledge_document: str = await self._knowledge_renderer.render_knowledge_document(self.gdb, lang)

        if knowledge_document:
            sep = "═" * 43
            knowledge_document = (
                f"{sep}\n"
                f"{loc['doc_header']}\n"
                f"{sep}\n"
                f"{knowledge_document}\n"
                f"{sep}"
            )
        else:
            knowledge_document = ""

        # 2. Full retrieval of everything CLAM knows about the user from LTM.
        # In addition to structured facts from GraphDB, we also retrieve
        # vector facts from ChromaDB for a complete picture.
        all_ltm_facts = await self.ltm.get_recent_semantic(limit=20)
        semantic_res = await self.ltm.search_semantic(user_prompt, n_results=3)

        # Combine general facts with prompt-specific ones
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
            facts = loc["no_observed"]
        context_block = f"\n--- {loc['observed_header']} ---\n{facts}\n-----------------------"

        # 3. Build the Unified System Prompt in the configured language
        system_prompt_template: str = get_clam_system_prompt(lang)
        system_prompt = system_prompt_template.format(
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
