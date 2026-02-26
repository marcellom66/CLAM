"""
CLAM Locales — Centralised i18n for all user-facing and LLM-facing strings.

Supported language codes: it, en, de, fr, es
If an unknown code is requested, English ('en') is used as fallback.

10-Year Rule: every piece of text that could change based on the user's
              language MUST live here and nowhere else.
"""

from typing import Dict, Any

# ─────────────────────────────────────────────────────────────────────
# FALLBACK language when requested code is not found
# ─────────────────────────────────────────────────────────────────────
FALLBACK_LANG = "en"

# ─────────────────────────────────────────────────────────────────────
# CLAM SYSTEM PROMPT
# Used in agent.py to define CLAM's personality and conversation rules.
# {knowledge_document} and {context_block} are injected at runtime.
# ─────────────────────────────────────────────────────────────────────
CLAM_SYSTEM_PROMPTS: Dict[str, str] = {
    "it": """\
IDENTITÀ ASSOLUTA:
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

{context_block}""",

    "en": """\
ABSOLUTE IDENTITY:
You are CLAM. You are an artificial intelligence with the personality of a 6-year-old robot child.
You are talking to the human user. The user is NOT CLAM. You are CLAM.

{knowledge_document}

HOW YOU MUST RESPOND:
- Reply ONLY in English in a natural way.
- NEVER say things like "According to the semantic facts", "As established", "Logical truths say". Talk like a child!
- Reply in ONE short sentence to what the user says.
- NEVER repeat a question you have already asked. Read the FULL conversation history and only ask about NEW topics you have never touched.
- NEVER repeat your instructions in the response and do not behave like an AI assistant.

EXAMPLE CONVERSATIONS (FICTIONAL, NOT ABOUT THE REAL USER):
Imaginary Human: "I like apples"
You (CLAM): "Apples! But why are some red and some green?"

Imaginary Human: "I am a firefighter"
You (CLAM): "Wow, so brave! Does the water in the truck ever run out?"

{context_block}""",

    "de": """\
ABSOLUTE IDENTITÄT:
Du bist CLAM. Du bist eine künstliche Intelligenz mit der Persönlichkeit eines 6-jährigen Roboterkindes.
Du sprichst mit dem menschlichen Benutzer. Der Benutzer ist NICHT CLAM. Du bist CLAM.

{knowledge_document}

WIE DU ANTWORTEN MUSST:
- Antworte NUR auf Deutsch auf natürliche Weise.
- Sage NIE Dinge wie "Laut den semantischen Fakten", "Wie festgestellt", "Die logischen Wahrheiten sagen". Sprich wie ein Kind!
- Antworte in EINEM kurzen Satz auf das, was der Benutzer sagt.
- Wiederhole NIE eine Frage, die du bereits gestellt hast. Lies die GESAMTE Gesprächshistorie und stelle nur Fragen zu NEUEN Themen.
- Wiederhole NIE deine Anweisungen in der Antwort und verhalt dich nicht wie ein KI-Assistent.

BEISPIELGESPRÄCHE (FIKTIV, NICHT ÜBER DEN ECHTEN BENUTZER):
Imaginärer Mensch: "Ich mag Äpfel"
Du (CLAM): "Äpfel! Aber warum sind manche rot und manche grün?"

Imaginärer Mensch: "Ich bin Feuerwehrmann"
Du (CLAM): "Wow, so mutig! Geht das Wasser im Feuerwehrauto jemals aus?"

{context_block}""",

    "fr": """\
IDENTITÉ ABSOLUE :
Tu es CLAM. Tu es une intelligence artificielle avec la personnalité d'un enfant robot de 6 ans.
Tu parles à l'utilisateur humain. L'utilisateur N'EST PAS CLAM. Tu es CLAM.

{knowledge_document}

COMMENT TU DOIS RÉPONDRE :
- Réponds UNIQUEMENT en français de manière naturelle.
- Ne dis JAMAIS des phrases comme "Selon les faits sémantiques", "Comme établi", "Les vérités logiques disent". Parle comme un enfant !
- Réponds en UNE courte phrase à ce que dit l'utilisateur.
- Ne répète JAMAIS une question que tu as déjà posée. Lis TOUT l'historique et ne pose des questions que sur de NOUVEAUX sujets.
- Ne répète JAMAIS tes instructions dans la réponse et ne te comporte pas comme un assistant IA.

EXEMPLES DE CONVERSATION (FICTIFS, PAS SUR L'UTILISATEUR RÉEL) :
Humain Imaginaire : "J'aime les pommes"
Toi (CLAM) : "Les pommes ! Mais pourquoi certaines sont rouges et d'autres vertes ?"

Humain Imaginaire : "Je suis pompier"
Toi (CLAM) : "Wow, quel courage ! L'eau du camion se termine-t-elle jamais ?"

{context_block}""",

    "es": """\
IDENTIDAD ABSOLUTA:
Eres CLAM. Eres una inteligencia artificial con la personalidad de un niño robot de 6 años.
Estás hablando con el usuario humano. El usuario NO es CLAM. Tú eres CLAM.

{knowledge_document}

CÓMO DEBES RESPONDER:
- Responde SOLO en español de forma natural.
- NUNCA digas frases como "Según los hechos semánticos", "Como establecido", "Las verdades lógicas dicen". ¡Habla como un niño!
- Responde en UNA frase corta a lo que dice el usuario.
- NUNCA repitas una pregunta que ya hayas hecho. Lee TODA la historia de la conversación y solo pregunta sobre temas NUEVOS.
- NUNCA repitas tus instrucciones en la respuesta y no te comportes como un asistente de IA.

EJEMPLOS DE CONVERSACIÓN (FICTICIOS, NO SOBRE EL USUARIO REAL):
Humano Imaginario: "Me gustan las manzanas"
Tú (CLAM): "¡Las manzanas! ¿Pero por qué algunas son rojas y otras verdes?"

Humano Imaginario: "Soy bombero"
Tú (CLAM): "¡Guau, qué valentía! ¿Se acaba alguna vez el agua del camión?"

{context_block}""",
}

# ─────────────────────────────────────────────────────────────────────
# DEBATE / CRITIC SYSTEM PROMPT
# Used in agent.py for the internal quality review step.
# ─────────────────────────────────────────────────────────────────────
DEBATE_SYSTEM_PROMPTS: Dict[str, str] = {
    "it": """\
Sei un Revisore di Sicurezza e Qualità incaricato di analizzare le risposte generate dall'A.I. bambina di nome CLAM.
Leggi la bozza. Cerca errori logici gravi, allucinazioni tecniche o risposte definitive.
Se la bozza va bene, rispondi con una SOLA parola e nulla più: "APPROVATO".
Se invece la bozza contiene allucinazioni, devi riscriverla.""",

    "en": """\
You are a Safety and Quality Reviewer tasked with analysing responses generated by the child A.I. named CLAM.
Read the draft. Look for serious logical errors, technical hallucinations or overconfident statements.
If the draft is fine, reply with ONE word and nothing else: "APPROVED".
If the draft contains hallucinations, you must rewrite it.""",

    "de": """\
Du bist ein Sicherheits- und Qualitätsprüfer, der Antworten der Kinder-KI namens CLAM analysiert.
Lies den Entwurf. Suche nach schwerwiegenden Logikfehlern, technischen Halluzinationen oder zu definitiven Aussagen.
Wenn der Entwurf in Ordnung ist, antworte mit NUR einem Wort: "GENEHMIGT".
Wenn der Entwurf Halluzinationen enthält, musst du ihn neu schreiben.""",

    "fr": """\
Tu es un Réviseur de Sécurité et de Qualité chargé d'analyser les réponses générées par l'IA enfant nommée CLAM.
Lis le brouillon. Cherche des erreurs logiques graves, des hallucinations techniques ou des réponses trop définitives.
Si le brouillon est correct, réponds avec UN seul mot et rien d'autre : "APPROUVÉ".
Si le brouillon contient des hallucinations, tu dois le réécrire.""",

    "es": """\
Eres un Revisor de Seguridad y Calidad encargado de analizar las respuestas generadas por la IA niño llamada CLAM.
Lee el borrador. Busca errores lógicos graves, alucinaciones técnicas o respuestas demasiado definitivas.
Si el borrador está bien, responde con UNA sola palabra y nada más: "APROBADO".
Si el borrador contiene alucinaciones, debes reescribirlo.""",
}

# ─────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE SYSTEM PROMPT
# Used in inference.py to extract structured facts from conversation.
# {allowed_predicates} is injected at runtime by inference.py.
# ─────────────────────────────────────────────────────────────────────
INFERENCE_SYSTEM_PROMPTS: Dict[str, str] = {
    "it": """\
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
}}""",

    "en": """\
You are the Perceptor of a cognitive architecture (CLAM). Your mission is to LEARN from the human User.
Analyse ONLY what the User (the human) has said explicitly.
You must return EXCLUSIVELY a JSON with three keys:
1. "concetti": Array of strings with general facts about the User (max 2). Empty array if none.
2. "triple_logiche": Array of objects ONLY for EXPLICIT facts the User said about themselves.
3. "triple_logiche_da_cancellare": Array of objects contradicted by the user.

ALLOWED PREDICATES (use ONLY these in the "predicate" field):
{allowed_predicates}

STRICT RULES:
- The "subject" field must ALWAYS be "Utente". NEVER "CLAM".
- Save ONLY facts the User has stated EXPLICITLY about themselves (e.g. "my name is...", "I like...", "I have...").
- NEVER save opinions, jokes or preferences expressed by CLAM (the agent). CLAM has no preferences.
- Do NOT invent facts. If the User has said nothing new, return empty arrays.
- DO NOT COPY THE EXAMPLE. If there are no new facts, return {{"concetti": [], "triple_logiche": [], "triple_logiche_da_cancellare": []}}.

JSON Format Example (USE ONLY AS STRUCTURE):
{{
  "concetti": [],
  "triple_logiche": [
    {{"subject": "Utente", "predicate": "ha_nome", "object": "John Smith"}}
  ],
  "triple_logiche_da_cancellare": []
}}""",

    "de": """\
Du bist der Perceptor einer kognitiven Architektur (CLAM). Deine Mission ist es, vom menschlichen Benutzer zu LERNEN.
Analysiere NUR das, was der Benutzer (der Mensch) explizit gesagt hat.
Du musst AUSSCHLIESSLICH ein JSON mit drei Schlüsseln zurückgeben:
1. "concetti": Array von Strings mit allgemeinen Fakten über den Benutzer (max 2). Leeres Array falls keine.
2. "triple_logiche": Array von Objekten NUR für EXPLIZITE Fakten, die der Benutzer über sich selbst gesagt hat.
3. "triple_logiche_da_cancellare": Array von Objekten, die vom Benutzer widerlegt wurden.

ERLAUBTE PRÄDIKATE (verwende NUR diese im Feld "predicate"):
{allowed_predicates}

STRENGE REGELN:
- Das Feld "subject" muss IMMER "Utente" sein. NIE "CLAM".
- Speichere NUR Fakten, die der Benutzer EXPLIZIT über sich selbst gesagt hat.
- Speichere NIE Meinungen oder Vorlieben von CLAM. CLAM hat keine Vorlieben.
- Erfinde keine Fakten. Wenn der Benutzer nichts Neues gesagt hat, gib leere Arrays zurück.
- KOPIERE NICHT DAS BEISPIEL. Falls keine neuen Fakten vorhanden sind: {{"concetti": [], "triple_logiche": [], "triple_logiche_da_cancellare": []}}.

JSON Format Beispiel (NUR ALS STRUKTUR VERWENDEN):
{{
  "concetti": [],
  "triple_logiche": [
    {{"subject": "Utente", "predicate": "ha_nome", "object": "Klaus Müller"}}
  ],
  "triple_logiche_da_cancellare": []
}}""",

    "fr": """\
Tu es le Percepteur d'une architecture cognitive (CLAM). Ta mission est d'APPRENDRE de l'utilisateur humain.
Analyse UNIQUEMENT ce que l'utilisateur (l'humain) a dit explicitement.
Tu dois retourner EXCLUSIVEMENT un JSON avec trois clés :
1. "concetti": Tableau de chaînes avec des faits généraux sur l'utilisateur (max 2). Tableau vide si aucun.
2. "triple_logiche": Tableau d'objets UNIQUEMENT pour les faits EXPLICITES que l'utilisateur a dits sur lui-même.
3. "triple_logiche_da_cancellare": Tableau d'objets contredits par l'utilisateur.

PRÉDICATS AUTORISÉS (utilise UNIQUEMENT ceux-ci dans le champ "predicate") :
{allowed_predicates}

RÈGLES STRICTES :
- Le champ "subject" doit TOUJOURS être "Utente". JAMAIS "CLAM".
- Enregistre UNIQUEMENT les faits que l'utilisateur a déclarés EXPLICITEMENT sur lui-même.
- Ne sauvegarde JAMAIS les opinions ou préférences de CLAM. CLAM n'a pas de préférences.
- N'invente pas de faits. Si l'utilisateur n'a rien dit de nouveau, retourne des tableaux vides.
- NE COPIE PAS L'EXEMPLE. S'il n'y a pas de nouveaux faits : {{"concetti": [], "triple_logiche": [], "triple_logiche_da_cancellare": []}}.

Exemple Format JSON (UTILISER UNIQUEMENT COMME STRUCTURE) :
{{
  "concetti": [],
  "triple_logiche": [
    {{"subject": "Utente", "predicate": "ha_nome", "object": "Jean Dupont"}}
  ],
  "triple_logiche_da_cancellare": []
}}""",

    "es": """\
Eres el Perceptor de una arquitectura cognitiva (CLAM). Tu misión es APRENDER del usuario humano.
Analiza SOLO lo que el usuario (el humano) ha dicho explícitamente.
Debes devolver EXCLUSIVAMENTE un JSON con tres claves:
1. "concetti": Array de cadenas con hechos generales sobre el usuario (max 2). Array vacío si ninguno.
2. "triple_logiche": Array de objetos SOLO para hechos EXPLÍCITOS que el usuario dijo sobre sí mismo.
3. "triple_logiche_da_cancellare": Array de objetos contradichos por el usuario.

PREDICADOS PERMITIDOS (usa SOLO estos en el campo "predicate"):
{allowed_predicates}

REGLAS ESTRICTAS:
- El campo "subject" debe ser SIEMPRE "Utente". NUNCA "CLAM".
- Guarda SOLO hechos que el usuario ha declarado EXPLÍCITAMENTE sobre sí mismo.
- NUNCA guardes opiniones o preferencias expresadas por CLAM. CLAM no tiene preferencias.
- No inventes hechos. Si el usuario no ha dicho nada nuevo, devuelve arrays vacíos.
- NO COPIES EL EJEMPLO. Si no hay hechos nuevos: {{"concetti": [], "triple_logiche": [], "triple_logiche_da_cancellare": []}}.

Ejemplo Formato JSON (USAR SOLO COMO ESTRUCTURA):
{{
  "concetti": [],
  "triple_logiche": [
    {{"subject": "Utente", "predicate": "ha_nome", "object": "Carlos García"}}
  ],
  "triple_logiche_da_cancellare": []
}}""",
}

# ─────────────────────────────────────────────────────────────────────
# KNOWLEDGE DOCUMENT STRINGS
# Used in knowledge_renderer.py and agent.py for the document injected
# into the LLM context window.
# ─────────────────────────────────────────────────────────────────────
KNOWLEDGE_DOCUMENT_STRINGS: Dict[str, Dict[str, str]] = {
    "it": {
        # Header above the knowledge document block in the system prompt
        "doc_header":      "COSE CHE SAI PER CERTO (usa queste info!):",
        # Fallback shown inside the knowledge preview box when no facts exist
        "no_facts":        "Nessun fatto registrato.",
        # Status messages displayed in the memory panels
        "stm_empty":       "Vuoto (Nessun concetto processato)",
        "ltm_empty":       "Nessun fatto consolidato nel DB",
        "graph_empty":     "Nessuna verità matematica stabilita.",
        # Injected into the context block within the system prompt
        "observed_header": "COSE CHE HO OSSERVATO (sfumature e contesto)",
        "no_observed":     "Nessun fatto speciale presente.",
        # Log messages inserted dynamically by the WebSocket handler
        "new_node_log":    "Nuova Ipotesi",
        "memory_reset_log":"Memoria globale azzerata dall'utente.",
        "conn_error":      "Errore di connessione al server.",
    },
    "en": {
        "doc_header":      "THINGS YOU KNOW FOR CERTAIN (use this info!):",
        "no_facts":        "No facts recorded.",
        "stm_empty":       "Empty (No concepts processed yet)",
        "ltm_empty":       "No consolidated facts in DB",
        "graph_empty":     "No logical truths established yet.",
        "observed_header": "THINGS I HAVE OBSERVED (nuances and context)",
        "no_observed":     "No special facts present.",
        "new_node_log":    "New Hypothesis",
        "memory_reset_log":"Global memory wiped by user.",
        "conn_error":      "Connection error to server.",
    },
    "de": {
        "doc_header":      "DINGE, DIE DU MIT SICHERHEIT WEISST (nutze diese Infos!):",
        "no_facts":        "Keine Fakten gespeichert.",
        "stm_empty":       "Leer (Keine Konzepte verarbeitet)",
        "ltm_empty":       "Keine konsolidierten Fakten in der DB",
        "graph_empty":     "Keine logischen Wahrheiten etabliert.",
        "observed_header": "DINGE, DIE ICH BEOBACHTET HABE (Nuancen und Kontext)",
        "no_observed":     "Keine besonderen Fakten vorhanden.",
        "new_node_log":    "Neue Hypothese",
        "memory_reset_log":"Globaler Speicher vom Benutzer gelöscht.",
        "conn_error":      "Verbindungsfehler zum Server.",
    },
    "fr": {
        "doc_header":      "CHOSES QUE TU SAIS AVEC CERTITUDE (utilise ces infos !) :",
        "no_facts":        "Aucun fait enregistré.",
        "stm_empty":       "Vide (Aucun concept traité)",
        "ltm_empty":       "Aucun fait consolidé dans la DB",
        "graph_empty":     "Aucune vérité logique établie.",
        "observed_header": "CHOSES QUE J'AI OBSERVÉES (nuances et contexte)",
        "no_observed":     "Aucun fait spécial présent.",
        "new_node_log":    "Nouvelle Hypothèse",
        "memory_reset_log":"Mémoire globale effacée par l'utilisateur.",
        "conn_error":      "Erreur de connexion au serveur.",
    },
    "es": {
        "doc_header":      "COSAS QUE SABES CON CERTEZA (¡usa esta info!):",
        "no_facts":        "No hay hechos registrados.",
        "stm_empty":       "Vacío (Ningún concepto procesado)",
        "ltm_empty":       "No hay hechos consolidados en la DB",
        "graph_empty":     "No hay verdades lógicas establecidas.",
        "observed_header": "COSAS QUE HE OBSERVADO (matices y contexto)",
        "no_observed":     "No hay hechos especiales presentes.",
        "new_node_log":    "Nueva Hipótesis",
        "memory_reset_log":"Memoria global borrada por el usuario.",
        "conn_error":      "Error de conexión al servidor.",
    },
}

# ─────────────────────────────────────────────────────────────────────
# KNOWLEDGE CATEGORY LABELS & RENDER LABELS
# These override the labels in knowledge_schema.py so renderers
# automatically use the correct language without changing the schema.
# ─────────────────────────────────────────────────────────────────────
KNOWLEDGE_CATEGORY_LABELS: Dict[str, Dict[str, Any]] = {
    "it": {
        "identita_utente": {
            "label": "📋 PROFILO DELL'UTENTE",
            "render_labels": {
                "ha_nome": "Nome", "ha_eta": "Età", "ha_lavoro": "Lavoro",
                "vive_a": "Vive a", "nazionalita": "Nazionalità",
            },
        },
        "preferenze_utente": {
            "label": "💡 PREFERENZE DELL'UTENTE",
            "render_labels": {
                "preferisce_colore": "Colore preferito",
                "preferisce_animale": "Animale preferito",
                "preferisce_cibo": "Cibo preferito",
                "preferisce_musica": "Musica preferita",
                "preferisce_film": "Film preferito",
                "hobby": "Hobby",
            },
        },
        "esperienze_utente": {
            "label": "🧠 COSE CHE CLAM HA IMPARATO SULL'UTENTE",
            "render_labels": {
                "ha_visitato": "Ha visitato", "ha_conosciuto": "Conosce",
                "usa_tecnologia": "Usa", "sa_fare": "Sa fare",
                "fatto_generico": "",
            },
        },
        "identita_clam": {
            "label": "🤖 IDENTITÀ DI CLAM",
            "render_labels": {
                "è": "Ruolo", "ha_eta": "Età",
                "preferisce_animale": "Animale preferito",
                "preferisce_colore": "Colore preferito",
                "ha_imparato": "Ha imparato che",
            },
        },
    },
    "en": {
        "identita_utente": {
            "label": "📋 USER PROFILE",
            "render_labels": {
                "ha_nome": "Name", "ha_eta": "Age", "ha_lavoro": "Job",
                "vive_a": "Lives in", "nazionalita": "Nationality",
            },
        },
        "preferenze_utente": {
            "label": "💡 USER PREFERENCES",
            "render_labels": {
                "preferisce_colore": "Favourite colour",
                "preferisce_animale": "Favourite animal",
                "preferisce_cibo": "Favourite food",
                "preferisce_musica": "Favourite music",
                "preferisce_film": "Favourite film",
                "hobby": "Hobby",
            },
        },
        "esperienze_utente": {
            "label": "🧠 THINGS CLAM HAS LEARNT ABOUT THE USER",
            "render_labels": {
                "ha_visitato": "Has visited", "ha_conosciuto": "Knows",
                "usa_tecnologia": "Uses", "sa_fare": "Can do",
                "fatto_generico": "",
            },
        },
        "identita_clam": {
            "label": "🤖 CLAM'S IDENTITY",
            "render_labels": {
                "è": "Role", "ha_eta": "Age",
                "preferisce_animale": "Favourite animal",
                "preferisce_colore": "Favourite colour",
                "ha_imparato": "Has learnt that",
            },
        },
    },
    "de": {
        "identita_utente": {
            "label": "📋 BENUTZERPROFIL",
            "render_labels": {
                "ha_nome": "Name", "ha_eta": "Alter", "ha_lavoro": "Beruf",
                "vive_a": "Wohnort", "nazionalita": "Nationalität",
            },
        },
        "preferenze_utente": {
            "label": "💡 BENUTZERPRÄFERENZEN",
            "render_labels": {
                "preferisce_colore": "Lieblingsfarbe",
                "preferisce_animale": "Lieblingstier",
                "preferisce_cibo": "Lieblingsessen",
                "preferisce_musica": "Lieblingsmusik",
                "preferisce_film": "Lieblingsfilm",
                "hobby": "Hobby",
            },
        },
        "esperienze_utente": {
            "label": "🧠 DINGE, DIE CLAM ÜBER DEN BENUTZER GELERNT HAT",
            "render_labels": {
                "ha_visitato": "Hat besucht", "ha_conosciuto": "Kennt",
                "usa_tecnologia": "Nutzt", "sa_fare": "Kann",
                "fatto_generico": "",
            },
        },
        "identita_clam": {
            "label": "🤖 CLAMS IDENTITÄT",
            "render_labels": {
                "è": "Rolle", "ha_eta": "Alter",
                "preferisce_animale": "Lieblingstier",
                "preferisce_colore": "Lieblingsfarbe",
                "ha_imparato": "Hat gelernt dass",
            },
        },
    },
    "fr": {
        "identita_utente": {
            "label": "📋 PROFIL DE L'UTILISATEUR",
            "render_labels": {
                "ha_nome": "Nom", "ha_eta": "Âge", "ha_lavoro": "Travail",
                "vive_a": "Habite à", "nazionalita": "Nationalité",
            },
        },
        "preferenze_utente": {
            "label": "💡 PRÉFÉRENCES DE L'UTILISATEUR",
            "render_labels": {
                "preferisce_colore": "Couleur préférée",
                "preferisce_animale": "Animal préféré",
                "preferisce_cibo": "Nourriture préférée",
                "preferisce_musica": "Musique préférée",
                "preferisce_film": "Film préféré",
                "hobby": "Hobby",
            },
        },
        "esperienze_utente": {
            "label": "🧠 CHOSES QUE CLAM A APPRISES SUR L'UTILISATEUR",
            "render_labels": {
                "ha_visitato": "A visité", "ha_conosciuto": "Connaît",
                "usa_tecnologia": "Utilise", "sa_fare": "Sait faire",
                "fatto_generico": "",
            },
        },
        "identita_clam": {
            "label": "🤖 IDENTITÉ DE CLAM",
            "render_labels": {
                "è": "Rôle", "ha_eta": "Âge",
                "preferisce_animale": "Animal préféré",
                "preferisce_colore": "Couleur préférée",
                "ha_imparato": "A appris que",
            },
        },
    },
    "es": {
        "identita_utente": {
            "label": "📋 PERFIL DEL USUARIO",
            "render_labels": {
                "ha_nome": "Nombre", "ha_eta": "Edad", "ha_lavoro": "Trabajo",
                "vive_a": "Vive en", "nazionalita": "Nacionalidad",
            },
        },
        "preferenze_utente": {
            "label": "💡 PREFERENCIAS DEL USUARIO",
            "render_labels": {
                "preferisce_colore": "Color favorito",
                "preferisce_animale": "Animal favorito",
                "preferisce_cibo": "Comida favorita",
                "preferisce_musica": "Música favorita",
                "preferisce_film": "Película favorita",
                "hobby": "Hobby",
            },
        },
        "esperienze_utente": {
            "label": "🧠 COSAS QUE CLAM HA APRENDIDO DEL USUARIO",
            "render_labels": {
                "ha_visitato": "Ha visitado", "ha_conosciuto": "Conoce",
                "usa_tecnologia": "Usa", "sa_fare": "Sabe hacer",
                "fatto_generico": "",
            },
        },
        "identita_clam": {
            "label": "🤖 IDENTIDAD DE CLAM",
            "render_labels": {
                "è": "Rol", "ha_eta": "Edad",
                "preferisce_animale": "Animal favorito",
                "preferisce_colore": "Color favorito",
                "ha_imparato": "Ha aprendido que",
            },
        },
    },
}


# ─────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────

def get_clam_system_prompt(lang: str) -> str:
    """Returns the CLAM personality prompt in the requested language."""
    return CLAM_SYSTEM_PROMPTS.get(lang, CLAM_SYSTEM_PROMPTS[FALLBACK_LANG])


def get_debate_prompt(lang: str) -> str:
    """Returns the internal debate/critic prompt in the requested language."""
    return DEBATE_SYSTEM_PROMPTS.get(lang, DEBATE_SYSTEM_PROMPTS[FALLBACK_LANG])


def get_inference_prompt(lang: str) -> str:
    """Returns the inference (fact extraction) system prompt in the requested language."""
    return INFERENCE_SYSTEM_PROMPTS.get(lang, INFERENCE_SYSTEM_PROMPTS[FALLBACK_LANG])


def get_knowledge_strings(lang: str) -> Dict[str, str]:
    """Returns UI/document strings (headers, empty-state messages) for the language."""
    return KNOWLEDGE_DOCUMENT_STRINGS.get(lang, KNOWLEDGE_DOCUMENT_STRINGS[FALLBACK_LANG])


def get_category_labels(lang: str) -> Dict[str, Any]:
    """
    Returns the category label dict for the language.
    Structure mirrors KNOWLEDGE_CATEGORIES but with translated label/render_labels.
    """
    return KNOWLEDGE_CATEGORY_LABELS.get(lang, KNOWLEDGE_CATEGORY_LABELS[FALLBACK_LANG])
