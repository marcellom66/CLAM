"""
Knowledge Renderer: trasforma le triple raw del GraphDB
in un documento strutturato e leggibile per l'LLM.

Principio architetturale: un LLM non capisce "Utente -> ha_nome -> Marcello".
Capisce molto meglio:
  📋 PROFILO DELL'UTENTE:
    Nome: Marcello Mangione
    Età: 60 anni

Questo modulo è il ponte tra il database (struttura dati)
e la context window dell'LLM (linguaggio naturale).
"""

from typing import Dict, List, Set, Tuple
from clam.memory.graph_db import GraphDB
from clam.core.models import LogicalTriple
from clam.core.knowledge_schema import (
    KNOWLEDGE_CATEGORIES,
    normalize_predicate,
)


class KnowledgeRenderer:
    """
    Genera un documento di conoscenza strutturato per categorie ontologiche.
    Il documento viene iniettato nel system prompt dell'LLM al posto
    della vecchia lista piatta di triple.
    """

    async def render_knowledge_document(self, graph_db: GraphDB) -> str:
        """
        Pipeline completa: legge tutte le triple, le normalizza,
        le organizza per categorie e genera il documento in linguaggio naturale.

        Returns:
            Stringa vuota se non ci sono fatti, altrimenti il documento strutturato.
        """
        all_triples: List[LogicalTriple] = await graph_db.get_all_triples()

        if not all_triples:
            return ""

        # 1. Normalizza i predicati e de-duplica per contenuto semantico
        normalized: List[LogicalTriple] = self._normalize_and_deduplicate(all_triples)

        # 2. Raggruppa per categoria ontologica
        categorized: Dict[str, List[LogicalTriple]] = self._categorize_triples(normalized)

        # 3. Renderizza in formato documento naturale
        document: str = self._render_document(categorized)

        return document

    def _normalize_and_deduplicate(
        self, triples: List[LogicalTriple]
    ) -> List[LogicalTriple]:
        """
        Normalizza i predicati e rimuove duplicati semantici.
        Due triple sono duplicate se hanno stesso (subject, predicate_normalizzato, object_).
        In caso di duplicati, tiene quella con confidence più alto.
        """
        # Mappa: (subject_lower, pred_normalizzato, object_lower) → tripla migliore
        best_triples: Dict[Tuple[str, str, str], LogicalTriple] = {}

        for triple in triples:
            normalized_pred: str = normalize_predicate(triple.predicate)

            # Chiave di de-duplicazione case-insensitive
            dedup_key: Tuple[str, str, str] = (
                triple.subject.strip().lower(),
                normalized_pred,
                triple.object_.strip().lower(),
            )

            existing = best_triples.get(dedup_key)
            if existing is None or triple.confidence > existing.confidence:
                # Creiamo una copia con il predicato normalizzato
                best_triples[dedup_key] = LogicalTriple(
                    id_tripla=triple.id_tripla,
                    subject=triple.subject,
                    predicate=normalized_pred,
                    object_=triple.object_,
                    confidence=triple.confidence,
                    timestamp=triple.timestamp,
                )

        return list(best_triples.values())

    def _categorize_triples(
        self, triples: List[LogicalTriple]
    ) -> Dict[str, List[LogicalTriple]]:
        """
        Smista ogni tripla nella categoria ontologica giusta
        basandosi su (entity del soggetto + predicato).
        Le triple orfane (predicato non riconosciuto) finiscono in 'esperienze_utente'.
        """
        categorized: Dict[str, List[LogicalTriple]] = {
            cat_name: [] for cat_name in KNOWLEDGE_CATEGORIES
        }

        for triple in triples:
            placed: bool = False
            subject_lower: str = triple.subject.strip().lower()

            for cat_name, cat_data in KNOWLEDGE_CATEGORIES.items():
                entity_lower: str = cat_data["entity"].lower()

                # Match: il soggetto corrisponde all'entità della categoria
                # E il predicato è tra quelli ammessi
                if (
                    subject_lower == entity_lower
                    and triple.predicate in cat_data["predicates"]
                ):
                    categorized[cat_name].append(triple)
                    placed = True
                    break

            # Fallback: le triple non classificate vanno nelle esperienze
            if not placed:
                categorized["esperienze_utente"].append(triple)

        return categorized

    def _render_document(
        self, categorized: Dict[str, List[LogicalTriple]]
    ) -> str:
        """
        Genera il documento finale in linguaggio naturale strutturato.

        Le categorie 'identità' e 'preferenze' vengono renderizzate
        come key-value (Nome: Marcello).
        Le categorie 'esperienze' vengono renderizzate come lista puntata.
        """
        sections: List[str] = []

        for cat_name, cat_data in KNOWLEDGE_CATEGORIES.items():
            triples: List[LogicalTriple] = categorized.get(cat_name, [])

            if not triples:
                continue

            render_labels: Dict[str, str] = cat_data.get("render_labels", {})
            label: str = cat_data["label"]

            # Distinguiamo tra categorie "key-value" e categorie "lista"
            if cat_name in ("esperienze_utente",):
                # Formato lista puntata per le esperienze
                lines: List[str] = [f"{label}:"]
                for t in triples:
                    prefix: str = render_labels.get(t.predicate, "")
                    if prefix:
                        lines.append(f"  - {prefix} {t.object_}")
                    else:
                        lines.append(f"  - {t.object_}")
                sections.append("\n".join(lines))
            else:
                # Formato key-value per identità e preferenze
                lines = [f"{label}:"]
                # Raggruppa per predicato per evitare duplicati visivi
                seen_predicates: Set[str] = set()
                for t in triples:
                    if t.predicate in seen_predicates:
                        continue
                    seen_predicates.add(t.predicate)

                    readable_label: str = render_labels.get(
                        t.predicate, t.predicate.replace("_", " ").title()
                    )
                    lines.append(f"  {readable_label}: {t.object_}")
                sections.append("\n".join(lines))

        if not sections:
            return ""

        return "\n\n".join(sections)
