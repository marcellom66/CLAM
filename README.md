# CLAM — Cognitive Large Application Model

CLAM is an LLM-based agent with a two-tier memory architecture. The goal is **not** to fine-tune the model, but to pair it with a dynamic database system that simulates human cognitive processes: perception, short-term evaluation, validation through confirmation, long-term consolidation, and **memory decay** (forgetting) for unconfirmed information.

For a deep-dive into the full architecture, design rules and overall vision, see **[CLAM_Architecture.md](CLAM_Architecture.md)**.

---

## ✨ Key Features

### Memory System
- **Short-Term Buffer (Scratchpad):** High-speed volatile memory (SQLite in-memory) for real-time concept processing.
- **Long-Term Memory (Bifurcated):** Persistent vector store split into two distinct collections:
  - **Semantic Memory (Facts):** Objective data and static user preferences.
  - **Episodic Memory (Experiences):** Traces of past interactions, past decisions, and historical problem-solving to avoid repeating mistakes.
- **GraphDB (Knowledge Graph):** A relational triple store (`subject → predicate → object`) that holds the agent's structured, factual knowledge about users and the environment.

### Cognitive Engines (Async Workers)
- **Inference Engine (The Perceiver):** Analyses user prompts and generated responses in the background, extracting new dedductions and writing them to the Short-Term Buffer and Knowledge Graph.
- **The Critic (High-Scepticism Validator):** Scans the Short-Term Buffer using a *devil's advocate* logic (Reflexion-inspired). It doesn't look for confirmations — it looks for contradictions.
- **Garbage Collector & Memory Evolution:**
  - **Promotion & Zettelkasten Linking:** Nodes with `confidence_score >= PROMOTION_THRESHOLD` are promoted to Long-Term Memory with vector-link generation to related existing nodes.
  - **Decay (Forgetting):** Nodes idle beyond `DECAY_TIME` with a low confidence score are permanently deleted — making room for better information.

### Self-Reflection (Advanced Auto-Correction Loop)
- **Familiarity Check:** For simple requests, the agent retrieves facts from Semantic Memory for a fast-track response.
- **Recollection:** For complex requests, a deep search through Episodic Memory reconstructs relevant historical event chains.
- **Internal Debate:** Before responding, a secondary LLM call (with an antagonistic role) evaluates the draft for violations of learned rules and forces a rewrite if errors are found.

### Structured Knowledge Graph API
A full REST API to inspect and manage the Knowledge Graph directly:
- `GET /api/knowledge/document` — Returns the structured knowledge document as the LLM sees it.
- `POST /api/knowledge/triple` — Manually adds a triple with automatic predicate normalisation.
- `DELETE /api/knowledge/triple/{id}` — Surgically removes a single triple.
- `POST /api/knowledge/seed` — Loads (or reloads) foundational facts from `seed_truths.yaml`.
- `POST /api/knowledge/reset-and-seed` — Full memory reset and seed reload.

### Seed Truths System
A `seed_truths.yaml` file allows you to pre-load fundamental, immutable facts (user identity, preferences, context) at startup. The seed is loaded **only once** when the Knowledge Graph is empty, preventing data duplication on restarts.

### Multilanguage Support
The Knowledge Renderer and the UI support multiple interface languages:
- 🇮🇹 Italian (default)
- 🇬🇧 English
- 🇩🇪 German
- 🇫🇷 French
- 🇪🇸 Spanish

The language can be changed from the dashboard without restarting the server.

### Real-Time Dashboard (WebSocket)
A completely decoupled frontend to observe the agent "thinking" and evolving in real time:
- **Stream of Consciousness:** A scrolling terminal showing Critic and Internal Debate logs.
- **Bifurcated Memory Dashboard:** Separate counters for Volatile Nodes (STM), Semantic Memory (Facts), Episodic Memory (Experiences), and Knowledge Graph triples.
- **Zettelkasten Growth Graph:** Interactive node graph using Vis.js. Volatile nodes pulse; promoted and linked nodes form visible clusters; decaying nodes fade out.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally with `qwen2.5:3b` (or any model configured in `config.yaml`)
- Dependencies listed in `requirements.txt`

---

## Installation & Setup

```bash
# 1. Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Edit seed_truths.yaml with your own identity facts

# 4. Start the CLAM system
./run_clam.sh
```

Then open your browser at `http://localhost:8000`.

---

## Project Structure

```
CLAM/
├── clam/
│   ├── api/           # FastAPI server (REST + WebSocket endpoints)
│   ├── core/          # Agent, KnowledgeRenderer, KnowledgeSchema, Models
│   ├── engines/       # InferenceEngine, Critic, GarbageCollector
│   ├── memory/        # ShortTermBuffer, LongTermMemory, GraphDB
│   └── llm/           # Ollama LLM client
├── data/              # Persistent databases (ChromaDB, SQLite)
├── index.html         # Dashboard frontend
├── seed_truths.yaml   # Foundational facts pre-loaded at startup
├── config.yaml        # All thresholds, timeouts, model settings
└── run_clam.sh        # Startup script
```

---

## Academic References

- **MemGPT** — *Towards LLMs as Operating Systems* (Packer et al.)
- **CoALA** — *Cognitive Architectures for Language Agents* (Sumers et al.)
- **Reflexion** — *Language Agents with Verbal Reinforcement Learning* (Shinn et al.)
- **Self-Refine** (Madaan et al.)
- **Generative Agents** — *Interactive Simulacra of Human Behavior* (Park et al.)

---

*Repository created for visualisation and tracking of the CLAM project.*
