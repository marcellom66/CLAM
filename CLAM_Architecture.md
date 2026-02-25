# CLAM Architecture

## 1. Visione Generale
Il sistema CLAM è un agente basato su LLM (Large Language Model) dotato di un'architettura di memoria a due livelli. L'obiettivo non è fare fine-tuning del modello, ma affiancargli un sistema di database dinamico che simuli il processo cognitivo umano: percezione, valutazione temporanea, validazione tramite conferme, consolidamento a lungo termine e, fondamentale, il "decadimento" (l'oblio) per le informazioni non confermate.
Il sistema esporrà anche un'interfaccia visiva (Dashboard) disaccoppiata per monitorare il flusso di coscienza e l'evoluzione dei nodi di memoria.

## 2. Fase 1: Struttura Dati e Memorie
**Direttiva per Antigravity:** Evitare assolutamente dizionari Python non tipizzati. Utilizzare Pydantic o dataclasses per garantire la rigidità dei dati. Un dato corrotto nel buffer distruggerà i ragionamenti futuri.

### A. Short-Term Buffer (La Memoria Volatile / Scratchpad)
Un database in RAM (es. SQLite in-memory o Redis) ad altissima velocità per l'elaborazione dei concetti in tempo reale.
*Struttura del dato (MemoryNode):*
* `id_concetto` (UUID)
* `descrizione` (String)
* `confidence_score` (Int, default: 1)
* `timestamp_creazione` (Datetime)
* `timestamp_ultimo_accesso` (Datetime)
* `contesto_origine` (String - dove e come ha dedotto l'informazione)

### B. Long-Term Memory (Il Database Consolidato e Biforcato)
Il deposito permanente. Cruciale: Sulla base delle architetture cognitive avanzate (rif. CoALA), il database permanente deve essere diviso in due collezioni vettoriali distinte:
* **Memoria Semantica (I Fatti):** Dati oggettivi e preferenze statiche (es. "L'utente usa ESP32", "L'indirizzo IP del server è 192.168.1.10").
* **Memoria Episodica (Le Esperienze):** Tracce di interazioni passate, processi decisionali e risoluzione di problemi (es. "Il 12 Maggio abbiamo risolto un bug sul sensore capacitivo filtrando il rumore con una media mobile"). Questa è la base per non ripetere gli stessi errori.

## 3. Fase 2: I Motori Logici (Worker Asincroni)
**Direttiva per Antigravity:** Questi moduli devono girare in modo indipendente. È obbligatorio l'uso di asyncio per non bloccare il loop principale.

### A. Inference Engine (Il Percettore)
Analizza il prompt dell'utente e la risposta generata. Genera in background un JSON strutturato con le nuove "deduzioni" da inserire nello Short-Term Buffer.

### B. The Critic (Il Motore di Validazione ad Alto Scetticismo)
Funzione: Quando l'LLM affronta un argomento, il Critic scansiona lo Short-Term Buffer.
Logica "Avvocato del Diavolo" (rif. Reflexion): Non deve cercare conferme, ma contraddizioni. Il prompt di sistema del Critic deve obbligarlo a trovare falle nel ragionamento temporaneo attraverso un dibattito interno testuale, non solo matematico.
Aggiornamento Score: Se il concetto resiste al Critic, `confidence_score` += 1. Se fallisce, `confidence_score` -= 1.

### C. Garbage Collector & Memory Evolution (Promozione, Collegamento e Oblio)
Un task asincrono che cicla sul buffer ogni N minuti.
* **Promozione e Zettelkasten (Link Generation):** Se `confidence_score >= PROMOTION_THRESHOLD`, il nodo viene estratto. PRIMA di salvarlo nella Long-Term Memory, il sistema esegue una ricerca vettoriale per trovare nodi simili già esistenti e crea dei metadati di collegamento (Linkage). Le memorie non sono isolate, formano un grafo.
* **Decadimento (L'Oblio):** Se `(Time.now() - timestamp_ultimo_accesso) > DECAY_TIME` e `confidence_score < MIN_SCORE`, il nodo viene cancellato permanentemente (implementazione della vulnerabilità cognitiva: fare spazio per concetti migliori).

## 4. Fase 3: Il Loop di Autocorrezione (Self-Reflection Avanzato)
L'autocorrezione pura fallisce spesso a causa del bias di conferma dell'LLM. Antigravity deve implementare una logica di Recollection-Familiarity.
* **Valutazione della Familiarità (Familiarity Check):** L'LLM valuta se la richiesta dell'utente è basilare. Se sì, recupera i fatti dalla Memoria Semantica ed emette la risposta (Fast Track).
* **Ricostruzione del Ricordo (Recollection):** Se la richiesta è complessa, innesca una ricerca profonda nella Memoria Episodica per ricostruire le catene di eventi simili del passato.
* **Dibattito Interno (Internal Debate):** Prima di rispondere, una chiamata secondaria all'LLM (con un ruolo antagonistico) valuta la bozza alla ricerca di violazioni delle regole apprese. Se trova errori, forza la riscrittura.

## 5. Fase 4: La Dashboard CLAM (L'Interfaccia Visiva)
Un modulo frontend totalmente disaccoppiato per osservare l'agente "pensare" ed evolvere.
* **Backend (API Bridge):** Tecnologia: FastAPI con WebSockets (VIETATO IL POLLING). Il motore logico invierà aggiornamenti push al frontend solo ai cambi di stato delle memorie.
* **Frontend (L'Interfaccia):** Tecnologia Prototipo: Streamlit, Gradio, o HTML/JS con librerie per grafi (es. Vis.js).
* **Elementi Visivi Richiesti:**
  * Stream of Consciousness: Un terminale a scorrimento (log del Critic e del Dibattito Interno).
  * Dashboard Memoria Biforcata: Contatori separati per Nodi Volatili, Memoria Semantica (Fatti) e Memoria Episodica (Esperienze).
  * Il Grafo di Crescita Zettelkasten: Visualizzazione a nodi interconnessi. I nodi volatili pulsano; i nodi promossi e collegati (Link Generation) formano cluster evidenti. I nodi in decadimento sbiadiscono.

## 6. Regole Architetturali Ferree (Critical Warnings)
* **Nessun Hardcoding:** Soglie di promozione, tempi di decadimento, dimensioni dei chunk vettoriali e chiavi API devono risiedere in un file `config.yaml`.
* **Concurrency Locks:** Implementare rigorosamente i Lock (`asyncio.Lock()`) per evitare "race conditions" durante le scritture asincrone sui buffer di memoria.
* **Gestione dei Costi Token:** La separazione tra Familiarity (risposta rapida) e Recollection (risposta profonda) è essenziale per non saturare la context window e abbattere i costi delle API (rif. MemGPT).

## 7. Riferimenti Accademici
* MemGPT (Towards LLMs as Operating Systems) (Packer et al.)
* Cognitive Architectures for Language Agents (CoALA) (Sumers et al.)
* Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al.) & Self-Refine (Madaan et al.)
* Generative Agents: Interactive Simulacra of Human Behavior (Park et al.)
