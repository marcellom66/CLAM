# CLAM Architecture

## 1. General Vision

The CLAM system is an LLM-based agent with a two-tier memory architecture. The goal is **not** to fine-tune the model, but to pair it with a dynamic database system that simulates the human cognitive process: perception, short-term evaluation, validation through confirmation, long-term consolidation, and — critically — **memory decay** (forgetting) for unconfirmed information.

The system also exposes a fully decoupled visual interface (Dashboard) to monitor the agent's stream of consciousness and the evolution of memory nodes.

---

## 2. Phase 1: Data Structures & Memories

> **Directive:** Absolutely avoid untyped Python dictionaries. Use Pydantic or dataclasses to guarantee data rigidity. Corrupted data in the buffer will destroy future reasoning chains.

### A. Short-Term Buffer (Volatile Memory / Scratchpad)

A high-speed in-memory database (SQLite in-memory) for real-time concept processing.

**Data structure (`MemoryNode`):**
| Field | Type | Description |
|---|---|---|
| `concept_id` | UUID | Unique identifier |
| `description` | String | The concept content |
| `confidence_score` | Int (default: 1) | Validation score |
| `creation_timestamp` | Datetime | When the node was created |
| `last_access_timestamp` | Datetime | When the node was last accessed |
| `origin_context` | String | Where and how the information was inferred |

### B. Long-Term Memory (Persistent & Bifurcated)

The permanent store. Based on advanced cognitive architectures (ref. CoALA), the permanent database is split into **two distinct vector collections**:

- **Semantic Memory (Facts):** Objective data and static user preferences (e.g. *"User uses ESP32"*, *"Server IP is 192.168.1.10"*).
- **Episodic Memory (Experiences):** Traces of past interactions, decision processes, and historical problem-solving (e.g. *"On 12 May we fixed a capacitive sensor bug by filtering noise with a moving average"*). This is the foundation for not repeating the same mistakes.

### C. GraphDB (Knowledge Graph)

A relational triple store that holds structured, factual knowledge as `(subject → predicate → object)` triples. This is the primary source of truth that the agent injects into its system prompt.

- Triples are normalised (predicate aliasing/canonicalisation) to prevent semantic duplicates.
- Triples are grouped into **ontological categories** (user identity, user preferences, user experiences) by the Knowledge Renderer.
- Pre-populated at startup via `seed_truths.yaml` when the graph is empty.

---

## 3. Phase 2: Logic Engines (Async Workers)

> **Directive:** These modules must run independently. `asyncio` is mandatory to avoid blocking the main event loop.

### A. Inference Engine (The Perceiver)

Analyses the user's prompt and the generated response. Generates in the background a structured JSON with new "deductions" to insert into the Short-Term Buffer and the Knowledge Graph.

### B. The Critic (High-Scepticism Validation Engine)

**Function:** When the LLM addresses a topic, the Critic scans the Short-Term Buffer.

**Devil's Advocate Logic** (ref. Reflexion): It must not seek confirmations — it must look for **contradictions**. The Critic's system prompt forces it to find weaknesses in the temporary reasoning through a textual internal debate, not just mathematical scoring.

**Score Update:**
- If the concept withstands the Critic → `confidence_score += 1`
- If the concept fails → `confidence_score -= 1`

> **Note:** The Critic is currently disabled for `qwen2.5:3b`, which always returns a negative verdict, blocking Ollama every 15 seconds. With the dual-write architecture, facts go directly to LTM without requiring the Critic. Re-enable with a larger model (e.g. `qwen3:8b` or above).

### C. Garbage Collector & Memory Evolution (Promotion, Linking & Decay)

An async task that loops over the Short-Term Buffer every N minutes.

- **Promotion & Zettelkasten (Link Generation):** If `confidence_score >= PROMOTION_THRESHOLD`, the node is extracted. Before saving to Long-Term Memory, the system runs a vector search to find similar existing nodes and creates linkage metadata. Memories are not isolated — they form a **knowledge graph**.
- **Decay (Forgetting):** If `(Time.now() - last_access_timestamp) > DECAY_TIME` and `confidence_score < MIN_SCORE`, the node is permanently deleted — implementing the *cognitive vulnerability* principle: making room for better concepts.

---

## 4. Phase 3: The Auto-Correction Loop (Advanced Self-Reflection)

Pure self-correction fails often due to the LLM's confirmation bias. CLAM implements a **Recollection-Familiarity** logic.

- **Familiarity Check:** The LLM evaluates whether the user's request is straightforward. If yes, it retrieves facts from Semantic Memory and issues the response (**Fast Track**).
- **Recollection:** If the request is complex, it triggers a deep search through Episodic Memory to reconstruct past chains of similar events.
- **Internal Debate:** Before responding, a secondary LLM call (with an antagonistic role) evaluates the draft for violations of learned rules. If it finds errors, it forces a rewrite.

---

## 5. Phase 4: The CLAM Dashboard (Visual Interface)

A fully decoupled frontend module to observe the agent "thinking" and evolving.

- **Backend (API Bridge):** FastAPI with WebSockets. **HTTP polling is strictly forbidden.** The logic engine sends push updates to the frontend only on memory state changes.
- **Frontend:** HTML/JavaScript with Vis.js for graph rendering.

**Required Visual Elements:**
- **Stream of Consciousness:** A scrolling terminal log of Critic and Internal Debate activity.
- **Bifurcated Memory Dashboard:** Separate counters for Volatile Nodes (STM), Semantic Memory (Facts), Episodic Memory (Experiences), and Knowledge Graph triples.
- **Zettelkasten Growth Graph:** Interactive node visualisation. Volatile nodes pulse; promoted and linked nodes form visible clusters; decaying nodes fade out.

---

## 6. Knowledge Renderer & Multilanguage Support

The **Knowledge Renderer** is the bridge between the raw database (data structures) and the LLM's context window (natural language). It converts raw triples into a structured, readable document that is injected into the system prompt.

For example, instead of:
```
User → has_name → Marcello
```
The LLM sees:
```
👤 USER PROFILE:
  Name: Marcello Mangione
  Age: 60
```

The renderer supports **5 languages** (Italian, English, German, French, Spanish) via localised category labels and predicate render labels, configurable from the dashboard without restarting the server.

---

## 7. Iron-Clad Architectural Rules (Critical Warnings)

- **No Hardcoding:** Promotion thresholds, decay times, vector chunk sizes, and API keys must reside in `config.yaml`.
- **Concurrency Locks:** Strictly implement locks (`asyncio.Lock()`) to avoid race conditions during async writes to memory buffers.
- **Token Cost Management:** The separation between Familiarity (fast response) and Recollection (deep response) is essential to avoid saturating the context window and to reduce API costs (ref. MemGPT).

---

## 8. Academic References

- **MemGPT** — *Towards LLMs as Operating Systems* (Packer et al.)
- **CoALA** — *Cognitive Architectures for Language Agents* (Sumers et al.)
- **Reflexion** — *Language Agents with Verbal Reinforcement Learning* (Shinn et al.)
- **Self-Refine** (Madaan et al.)
- **Generative Agents** — *Interactive Simulacra of Human Behavior* (Park et al.)
