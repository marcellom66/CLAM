# CLAM (Cognitive Large Application Model)

CLAM è un agente basato su LLM (Large Language Model) dotato di un'architettura di memoria a due livelli. L'obiettivo è quello di affiancare al modello un sistema di database dinamico che simuli il processo cognitivo umano attraverso le fasi di percezione, valutazione temporanea, validazione tramite conferme, consolidamento a lungo termine e "decadimento" (l'oblio) per le informazioni non confermate.

## Architettura di Sistema

Per i dettagli approfonditi su come è strutturata l'intera architettura, le regole di progettazione e la visione generale, consulta il documento **[CLAM_Architecture.md](CLAM_Architecture.md)**.

## Caratteristiche Principali

* **Short-Term Buffer (Scratchpad):** Una memoria volatile ad altissima velocità utilizzata per l'elaborazione dei concetti temporanei in tempo reale.
* **Long-Term Memory:** Il deposito permanente, ispirato ad architetture cognitive avanzate (es. CoALA), suddiviso in due livelli vettoriali:
  * **Memoria Semantica:** Fatti oggettivi e preferenze statiche dell'utente.
  * **Memoria Episodica:** Tracce di interazioni passate, processi decisionali pregressi e risoluzione di problemi storici per evitare la ripetizione di errori.
* **Inference Engine & The Critic:** Moduli worker asincroni. L'Inference Engine percepisce ed elabora, mentre il Critic agisce da "Avvocato del Divolo" per validare i concetti e arginare il bias di conferma tipico degli LLM.
* **Garbage Collector & Memory Evolution:** Un loop in background che promuove le memorie robuste collegandole (Zettelkasten) ed elimina le informazioni deboli e non verificate (Decadimento/Oblio).
* **Self-Reflection Avanzato (Loop di Autocorrezione):** Controlli di "Familiarità" e "Ricostruzione del Ricordo" (Recollection) seguiti da un dibattito interno prima di rispondere all'utente finale.

## Prerequisiti

- Python 3.10+
- Ollama (configurato e in esecuzione localmente, con `qwen2.5:3b` o altro modello configurato nel `config.yaml`)
- Dipendenze listate nel file `requirements.txt`

## Installazione ed Esecuzione

```bash
# 1. Crea e attiva l'ambiente virtuale
python3 -m venv .venv
source .venv/bin/activate

# 2. Installa le dipendenze
pip install -r requirements.txt

# 3. Avviare il sistema CLAM
./run_clam.sh
```

## Struttura del progetto
- `/clam/core`: Motori principali (Agent, Critic, Inference Engine).
- `/clam/memory`: Gestori della memoria a breve e lungo termine.
- `/clam/llm`: Client di connessione ai backend LLM (Ollama).
- `/data`: Database e istanze persistenti (es. chromadb, SQLite).

*(Repository generata per visualizzazione e tracking di CLAM)*
