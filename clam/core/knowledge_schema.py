"""
Ontologia Strutturata del Knowledge Graph di CLAM.

Definisce il vocabolario controllato di categorie e predicati ammessi
per impedire all'LLM di inventare predicati liberi e incoerenti.
L'Inference Engine DEVE usare solo i predicati definiti qui.

Principio: un LLM da 3B parametri capisce meglio un documento
strutturato in linguaggio naturale che una lista piatta di triple.

NB: le label localizzate (italiano, inglese, ecc.) vivono in locales.py.
    Qui rimane solo lo scheletro strutturale (entity + predicates)
    che è indipendente dalla lingua.
"""

from typing import Dict, List
from clam.core.locales import get_category_labels
from clam.config import CONFIG


# ─────────────────────────────────────────────────────────────────────
# KNOWLEDGE CATEGORIES (structural skeleton — language-independent)
# The 'label' and 'render_labels' fields are intentionally left blank here;
# they are fetched from locales.py at runtime via get_localized_categories().
# 'entity' and 'predicates' are internal ontology keys that never change.
# ─────────────────────────────────────────────────────────────────────
KNOWLEDGE_CATEGORIES: Dict[str, dict] = {
    "identita_utente": {
        "label": "",          # overridden by locales at runtime
        "predicates": ["ha_nome", "ha_eta", "ha_lavoro", "vive_a", "nazionalita"],
        "entity": "Utente",
        "render_labels": {},  # overridden by locales at runtime
    },
    "preferenze_utente": {
        "label": "",
        "predicates": [
            "preferisce_colore", "preferisce_animale", "preferisce_cibo",
            "preferisce_musica", "preferisce_film", "hobby"
        ],
        "entity": "Utente",
        "render_labels": {},
    },
    "esperienze_utente": {
        "label": "",
        "predicates": [
            "ha_visitato", "ha_conosciuto", "usa_tecnologia",
            "sa_fare", "fatto_generico"
        ],
        "entity": "Utente",
        "render_labels": {},
    },
    "identita_clam": {
        "label": "",
        "predicates": [
            "è", "ha_eta", "preferisce_animale", "preferisce_colore",
            "ha_imparato"
        ],
        "entity": "CLAM",
        "render_labels": {},
    },
}


def get_localized_categories(lang: str = None) -> Dict[str, dict]:
    """
    Returns KNOWLEDGE_CATEGORIES enriched with localized label and render_labels
    fetched from locales.py. If lang is None, uses CONFIG.language.

    The structural fields (entity, predicates) remain unchanged so that
    the rest of the codebase never needs to know about languages.
    """
    effective_lang: str = lang or CONFIG.language
    localized: Dict[str, dict] = get_category_labels(effective_lang)

    result: Dict[str, dict] = {}
    for cat_name, cat_data in KNOWLEDGE_CATEGORIES.items():
        merged = dict(cat_data)  # shallow copy of structural data
        if cat_name in localized:
            merged["label"] = localized[cat_name]["label"]
            merged["render_labels"] = localized[cat_name]["render_labels"]
        result[cat_name] = merged
    return result


# ─────────────────────────────────────────────────────────────────────
# NORMALIZZAZIONE DEI PREDICATI
# Mappa i predicati "liberi" che qwen2.5:3b tende a inventare
# verso i predicati standard definiti sopra.
# Espandibile nel tempo man mano che si scoprono nuove varianti.
# ─────────────────────────────────────────────────────────────────────
PREDICATE_NORMALIZATION: Dict[str, str] = {
    # Varianti del nome
    "nome": "ha_nome",
    "si_chiama": "ha_nome",
    "ha_name": "ha_nome",
    "name": "ha_nome",

    # Varianti dell'età
    "eta": "ha_eta",
    "età": "ha_eta",
    "age": "ha_eta",
    "anni": "ha_eta",

    # Varianti del lavoro
    "lavoro": "ha_lavoro",
    "professione": "ha_lavoro",
    "fa_il": "ha_lavoro",
    "lavora_come": "ha_lavoro",

    # Varianti delle preferenze animali
    "ha_animal_prefere": "preferisce_animale",
    "ha_preferenza_animale": "preferisce_animale",
    "animale_preferito": "preferisce_animale",
    "animal_preferito": "preferisce_animale",
    "ama_animale": "preferisce_animale",

    # Varianti delle preferenze colore
    "colore_preferito": "preferisce_colore",
    "ha_colore_preferito": "preferisce_colore",
    "ama_colore": "preferisce_colore",

    # Varianti delle preferenze cibo
    "cibo_preferito": "preferisce_cibo",
    "piatto_preferito": "preferisce_cibo",
    "ama_mangiare": "preferisce_cibo",

    # Varianti esperienze
    "è_stato_a": "ha_visitato",
    "ha_visto": "ha_visitato",
    "conosce": "ha_conosciuto",
    "usa": "usa_tecnologia",
    "utilizza": "usa_tecnologia",
    "è_usato_per": "usa_tecnologia",
    "è_tipologia_di": "fatto_generico",
}


def normalize_predicate(raw_predicate: str) -> str:
    """
    Converte un predicato libero generato dall'LLM nel predicato
    standard dell'ontologia. Se non lo trova, restituisce 'fatto_generico'.

    Motivazione: qwen2.5:3b inventa predicati a caso ad ogni inferenza.
    Senza normalizzazione, il DB si riempie di varianti incompatibili.
    """
    cleaned: str = raw_predicate.strip().lower()
    return PREDICATE_NORMALIZATION.get(cleaned, cleaned)


def get_allowed_predicates_prompt() -> str:
    """
    Genera la lista dei predicati ammessi in formato testo,
    da iniettare nel prompt dell'Inference Engine.
    """
    all_predicates: List[str] = []
    for category_data in KNOWLEDGE_CATEGORIES.values():
        all_predicates.extend(category_data["predicates"])

    # Rimuoviamo duplicati mantenendo l'ordine
    seen: set = set()
    unique: List[str] = []
    for pred in all_predicates:
        if pred not in seen:
            seen.add(pred)
            unique.append(pred)

    return ", ".join(unique)
