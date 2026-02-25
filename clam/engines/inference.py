import json
from clam.llm.ollama_client import OllamaClient
from clam.memory.short_term import ShortTermBuffer
from clam.memory.long_term import LongTermMemory
from clam.memory.graph_db import GraphDB
from clam.core.models import MemoryNode, VectorDBNode, LogicalTriple
from clam.core.knowledge_schema import normalize_predicate, get_allowed_predicates_prompt

# ─────────────────────────────────────────────────────────────────────
# PROMPT con predicati controllati dall'ontologia
# L'LLM DEVE usare SOLO i predicati ammessi nel campo "predicate"
# delle triple_logiche. Altrimenti il KnowledgeRenderer non sa dove metterle.
# ─────────────────────────────────────────────────────────────────────
INFERENCE_SYSTEM_PROMPT = """
Sei il Percettore di un'architettura cognitiva (CLAM). La tua missione è IMPARARE dall'Utente umano.
Analizza SOLO ciò che l'Utente (l'umano) ha detto esplicitamente.
Devi restituire ESCLUSIVAMENTE un JSON con tre chiavi:
1. "concetti": Array di stringhe con fatti generali sull'Utente (max 2). Array vuoto se nessuno.
2. "triple_logiche": Array di oggetti SOLO per fatti ESPLICITI detti dall'Utente su di sé.
3. "triple_logiche_da_cancellare": Array di oggetti smentiti dall'utente.

PREDICATI AMMESSI (usa SOLO questi nel campo "predicate"):
{allowed_predicates}

REGOLE RIGIDISSIME:
- Il campo "subject" deve essere SEMPRE "Utente". MAI "CLAM".
- Salva SOLO fatti che l'Utente ha detto ESPLICITAMENTE su di sé (es. "mi chiamo...", "mi piace...", "ho...").
- NON salvare MAI opinioni, battute o preferenze espresse da CLAM (l'agente). CLAM non ha preferenze.
- NON inventare fatti. Se l'Utente non ha detto nulla di nuovo, restituisci array vuoti.
- NON COPIARE L'ESEMPIO. Se non ci sono fatti nuovi, restituisci {{"concetti": [], "triple_logiche": [], "triple_logiche_da_cancellare": []}}.

Esempio Formato JSON (USA SOLO COME STRUTTURA):
{{
  "concetti": [],
  "triple_logiche": [
    {{"subject": "Utente", "predicate": "ha_nome", "object": "Mario Rossi"}}
  ],
  "triple_logiche_da_cancellare": []
}}
"""


class InferenceEngine:
    """Worker in background per la trasduzione dei concetti e del GraphRAG."""
    def __init__(self, llm_client: OllamaClient, stm: ShortTermBuffer, ltm: LongTermMemory, gdb: GraphDB):
        self.llm = llm_client
        self.stm = stm
        self.ltm = ltm
        self.gdb = gdb

    async def _save_fact(self, desc: str, source: str) -> None:
        """Salva un fatto semantico in STM e LTM."""
        node = MemoryNode(descrizione=desc, contesto_origine=source)
        await self.stm.add_node(node)
        
        lt_node = VectorDBNode(
            id_concetto=node.id_concetto,
            descrizione=desc,
            metadata={"contesto_origine": source, "direct_write": "true"}
        )
        try:
            await self.ltm.add_semantic_node(lt_node)
            print(f"[Inference Engine] 💾 Fatto salvato in LTM: {desc[:60]}")
        except Exception as e:
            print(f"[Inference Engine] LTM write fallito: {e}")

    async def _save_triple(self, sub: str, pred: str, obj: str) -> None:
        """
        Salva un fatto logico (GraphRAG) nel Triple Store.
        Il predicato viene normalizzato PRIMA del salvataggio per
        evitare varianti incoerenti nel Knowledge Graph.
        """
        # Normalizzazione: se l'LLM ha usato un predicato non standard,
        # lo mappiamo su quello corretto dell'ontologia
        normalized_pred: str = normalize_predicate(pred)
        if normalized_pred != pred:
            print(f"[Inference Engine] 🔄 Predicato normalizzato: '{pred}' → '{normalized_pred}'")

        triple = LogicalTriple(subject=sub, predicate=normalized_pred, object_=obj, confidence=5)
        await self.gdb.add_triple(triple)
        print(f"[Inference Engine] 🔗 Tripla salvata in GraphDB: [{sub} -> {normalized_pred} -> {obj}]")

    async def perceive(self, user_prompt: str, assistant_response: str) -> None:
        """Analizza la conversazione in background e salva/correla concetti & triple."""
        print(f"[Inference Engine] Avvio percezione in background...")
        
        # Recupero lo stato attuale dal Graph DB per permettere al LLM di capire se qualcosa è stato smentito
        user_triples = await self.gdb.get_triples_by_entity("Utente")
        clam_triples = await self.gdb.get_triples_by_entity("CLAM")
        all_triples = user_triples + clam_triples

        logical_truths = ""
        if all_triples:
            truths = "\n".join([f"- {t.subject} -> {t.predicate} -> {t.object_}" for t in all_triples])
            logical_truths = f"\n[VERITÀ ATTUALI NEL DB]\n{truths}\n(Se l'Utente smentisce o corregge una di queste verità, copiala ESATTAMENTE in 'triple_logiche_da_cancellare')\n"

        # Iniettiamo la lista dei predicati ammessi nel prompt di sistema
        system_prompt_with_predicates = INFERENCE_SYSTEM_PROMPT.format(
            allowed_predicates=get_allowed_predicates_prompt()
        )

        analysis_prompt = f"{logical_truths}\nUser: {user_prompt}\nAssistant: {assistant_response}\n\nEstrai i concetti, le triple da salvare e le triple da cancellare in JSON rigido."
        
        raw_json = await self.llm.generate_response(
            prompt=analysis_prompt,
            system_prompt=system_prompt_with_predicates,
            json_format=True
        )
        print(f"[Inference Engine] Raw JSON dedotto:\n{raw_json}")
        
        if not raw_json: return

        try:
            data = json.loads(raw_json)
            # Gestione array "concetti" (Vector & STM Memory)
            concetti = data.get("concetti", [])
            for c in concetti:
                if isinstance(c, str):
                    print(f"[Inference] Concetto vettoriale: {c}")
                    await self._save_fact(c, "Inference Perception")
            
            # Gestione array "triple_logiche" (Graph DB Memory)
            # FILTRO DI SICUREZZA: qwen2.5:3b tende a inventare fatti su CLAM
            # (es. "CLAM preferisce i cani") che non sono stati detti dall'utente.
            # Solo i fatti dal seed_truths.yaml possono avere subject "CLAM".
            triple = data.get("triple_logiche", [])
            for t in triple:
                sub = t.get("subject")
                pred = t.get("predicate")
                obj = t.get("object")
                if sub and pred and obj:
                    # Blocco hard: l'Inference Engine non può scrivere fatti su CLAM
                    if sub.strip().upper() == "CLAM":
                        print(f"[Inference Engine] 🚫 BLOCCATO: tripla su CLAM rifiutata [{sub} -> {pred} -> {obj}]")
                        continue
                    await self._save_triple(sub, pred, obj)
            
            # Gestione array "triple_logiche_da_cancellare" (Auto-Correzione)
            triple_del = data.get("triple_logiche_da_cancellare", [])
            for t in triple_del:
                sub = t.get("subject")
                pred = t.get("predicate")
                # Normalizziamo anche il predicato per la cancellazione
                if pred:
                    pred = normalize_predicate(pred)
                obj = t.get("object", t.get("object_"))  # Supporto fallback per l'underscore
                if sub and pred and obj:
                    await self.gdb.delete_triples_by_pattern(sub, pred, obj)
                    print(f"[Inference Engine] 🗑️ Tripla cancellata per Smentita: [{sub} -> {pred} -> {obj}]")

        except json.JSONDecodeError as e:
            print(f"[Inference Engine] Errore di decodifica JSON: {e}")
