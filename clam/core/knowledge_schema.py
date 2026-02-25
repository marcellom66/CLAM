"""
Ontologia Strutturata del Knowledge Graph di CLAM.

Definisce il vocabolario controllato di categorie e predicati ammessi
per impedire all'LLM di inventare predicati liberi e incoerenti.
L'Inference Engine DEVE usare solo i predicati definiti qui.

Principio: un LLM da 3B parametri capisce meglio un documento
strutturato in linguaggio naturale che una lista piatta di triple.
"""

from typing import Dict, List


# ─────────────────────────────────────────────────────────────────────
# CATEGORIE ONTOLOGICHE
# Ogni categoria raggruppa predicati logicamente affini.
# "label" = intestazione visibile nel documento generato per l'LLM
# "predicates" = predicati ammessi in questa categoria
# "entity" = soggetto a cui si riferiscono i predicati
# ─────────────────────────────────────────────────────────────────────
KNOWLEDGE_CATEGORIES: Dict[str, dict] = {
    "identita_utente": {
        "label": "📋 PROFILO DELL'UTENTE",
        "predicates": ["ha_nome", "ha_eta", "ha_lavoro", "vive_a", "nazionalita"],
        "entity": "Utente",
        # Formato di rendering: predicato → etichetta leggibile
        "render_labels": {
            "ha_nome": "Nome",
            "ha_eta": "Età",
            "ha_lavoro": "Lavoro",
            "vive_a": "Vive a",
            "nazionalita": "Nazionalità",
        }
    },
    "preferenze_utente": {
        "label": "💡 PREFERENZE DELL'UTENTE",
        "predicates": [
            "preferisce_colore", "preferisce_animale", "preferisce_cibo",
            "preferisce_musica", "preferisce_film", "hobby"
        ],
        "entity": "Utente",
        "render_labels": {
            "preferisce_colore": "Colore preferito",
            "preferisce_animale": "Animale preferito",
            "preferisce_cibo": "Cibo preferito",
            "preferisce_musica": "Musica preferita",
            "preferisce_film": "Film preferito",
            "hobby": "Hobby",
        }
    },
    "esperienze_utente": {
        "label": "🧠 COSE CHE CLAM HA IMPARATO SULL'UTENTE",
        "predicates": [
            "ha_visitato", "ha_conosciuto", "usa_tecnologia",
            "sa_fare", "fatto_generico"
        ],
        "entity": "Utente",
        # Le esperienze vengono renderizzate come lista, non key-value
        "render_labels": {
            "ha_visitato": "Ha visitato",
            "ha_conosciuto": "Conosce",
            "usa_tecnologia": "Usa",
            "sa_fare": "Sa fare",
            "fatto_generico": "",  # Nessun prefisso, testo libero
        }
    },
    "identita_clam": {
        "label": "🤖 IDENTITÀ DI CLAM",
        "predicates": [
            "è", "ha_eta", "preferisce_animale", "preferisce_colore",
            "ha_imparato"
        ],
        "entity": "CLAM",
        "render_labels": {
            "è": "Ruolo",
            "ha_eta": "Età",
            "preferisce_animale": "Animale preferito",
            "preferisce_colore": "Colore preferito",
            "ha_imparato": "Ha imparato che",
        }
    },
}


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
