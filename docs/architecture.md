# How Pikachu Works — Architecture

Pikachu is a **RAG (Retrieval-Augmented Generation)** system. Instead of relying on a pre-trained model to memorize facts about Sanchit, it retrieves relevant information at query time and feeds it as context to an LLM. This makes it accurate, updatable, and strictly grounded in real data.

---

## Request Flow

```
User Question
      │
      ▼
┌─────────────────────────────────────────┐
│           Flask API (/api/chat)          │
└─────────────────────────────────────────┘
      │
      ├─── 1. RETRIEVE ──────────────────────────────────────────────┐
      │         Augment query with recent conversation history        │
      │         Embed query with BAAI/bge-base-en-v1.5 (ONNX 768d)  │
      │         Search ChromaDB (cosine similarity)                   │
      │         Filter chunks below similarity threshold (0.3)        │
      │         Log each chunk + score for debugging                  │
      │         If nothing passes → tell LLM "no context found"       │
      │                                                              │
      ├─── 2. AUGMENT ───────────────────────────────────────────────┤
      │         Format as numbered, labelled context blocks           │
      │         Include similarity score per chunk                    │
      │         Append conversation history (last N turns)            │
      │         Apply system prompt (identity + strict grounding)    │
      │                                                              │
      └─── 3. GENERATE ──────────────────────────────────────────────┘
                LLM generates a response grounded ONLY in context
                Primary: Groq API (llama-3.3-70b-versatile)
                Fallback 1: Local Ollama (llama3.2:3b — offline capable)
                Fallback 2: HuggingFace API (optional, requires HF_API_TOKEN)
```

---

## Components

### 1. Knowledge Base

All facts about Sanchit live here as plain text:

| File | Contents | Chunking Strategy |
|------|----------|------------------|
| `data/knowledge_base/profile.md` | Education, career, skills, achievements | `##` section → 1 chunk |
| `data/knowledge_base/projects.md` | Detailed project descriptions | `##` section → 1 chunk |
| `data/knowledge_base/about_ai.md` | What this assistant is and how it was built | `##` section → 1 chunk |
| `data/knowledge_base/github.md` | Auto-fetched GitHub repos | `##` section → 1 chunk |
| `data/knowledge_base/Sanchit_CV.pdf` | Full CV — extracted as plain text | 500-char sliding window |
| `data/personal_data.json` | Hobbies, stories, opinions, FAQs | One document per structured entry |
| `data/website_pages/publications.json` | Research publications with abstracts | One document per publication |
| `data/website_pages/portfolios.json` | Portfolio projects with details | One document per ~500-char section |

### 2. Document Loader (`src/data/loader.py`)

Reads all knowledge files and prepares them for embedding. Three chunking strategies are used:

- **Markdown (`##` header splitting):** Each `##` section becomes one chunk. This keeps "Education", "Skills", "Publications List" etc. as coherent semantic units rather than character-sliced fragments. Sections > 1500 chars fall back to the sliding-window splitter.
- **JSON (natural structure):** Publications and portfolios are serialized per-entry (title + authors + journal + abstract for publications; title + tagline + impact + description + tech + challenge + solution for portfolios). Each entry is one chunk that stays together, so "what did GRILSS achieve?" retrieves the full GRILSS entry.
- **PDF (sliding window):** 500-char windows with 50-char overlap and sentence-boundary detection — used for the CV PDF where there are no semantic headers.

### 3. Embeddings (`src/rag/embeddings.py`)

Uses **fastembed** with `BAAI/bge-base-en-v1.5` (ONNX quantized, 768-dimensional vectors). This model significantly outperforms the older `all-MiniLM-L6-v2` (384-dim) on domain-specific retrieval with no GPU required — it runs on CPU via ONNX Runtime with ~90MB footprint.

Model cache:
- **Production (www-data):** `data/models/` (auto-downloaded on first request)
- **Developer builds (other users):** `~/.cache/fastembed/` (config.py detects write permission automatically)

### 4. Vector Store (`src/rag/vectorstore.py`)

**ChromaDB** stores the 250+ document embeddings persistently on disk in `data/processed/vectorstore/`. At query time, cosine similarity search returns the top-K chunks most semantically related to the user's question in milliseconds. Chunks below `SIMILARITY_THRESHOLD` (default 0.3) are filtered out before being returned.

### 5. Retriever (`src/rag/retriever.py`)

Before embedding, the raw user message goes through `build_retrieval_query()` which does two things:

**Query cleaning** — strips noise that degrades retrieval quality:
- Removes the person's name ("Is Sanchit Minocha...?" → already known topic)
- Removes question scaffolding ("What is his...", "Does he have...") that pulls the embedding toward generic identity chunks rather than topical content

**Selective history injection** — only for genuine follow-up questions, not self-contained ones. Prepending history to "What is his most impactful work?" injects topic noise; prepending it to "yes" or "other than that?" is essential. Two signals detect a follow-up:
1. **Dangling pronouns** — "it", "that", "those", "this", "the project", etc. with no self-contained topic
2. **Continuation phrases** — "yes", "sure", "tell me more", "go on", "what else?", "other than that?", etc.

When either signal fires, the retrieval query is prefixed with both the last user message *and* a snippet of the last assistant response. Including the assistant's prior answer pushes the embedding toward *complementary* content — so "other than that?" after an answer about FRIENDS and poetry retrieves Chess and Cooking rather than the same hobby chunks again.

**Similarity threshold filtering:** chunks below 0.3 cosine similarity are dropped. If no chunks pass, `format_context()` returns an explicit `"[No relevant information found in the knowledge base for this query.]"` string — so the LLM is directly told there is no context rather than silently receiving nothing.

**Debug logging:** every retrieval logs each chunk with its similarity score, source, and section label. Check `logs/app.log` to diagnose retrieval failures:
```
INFO  [1] score=0.8460 | projects.md › Publications | '## Publications  Sanchit has 15+ peer-reviewed...'
INFO  [2] score=0.7266 | publications.json › GRILSS | 'Publication: GRILSS: opening the gateway to...'
```

### 6. LLM Assistant (`src/llm/assistant.py`)

Takes the retrieved context + conversation history and sends them to an LLM with a strictly grounded system prompt:
- **Identity:** who it is, who built it, what it is NOT (not RAT 3.0, not GRILSS)
- **Grounding rules:** answer ONLY from context; if context is missing, say so — never hallucinate
- **Style:** warm, concise, human-like

**Accepted model values for the `model` API parameter:**

| Value | Backend | Notes |
|-------|---------|-------|
| `llama-3.3-70b-versatile` | Groq | Best quality — default |
| `llama-3.1-70b-versatile` | Groq | Good quality |
| `llama-3.1-8b-instant` | Groq | Fast, lower accuracy |
| `llama3-8b-8192` | Groq | Llama 3 8B on Groq |
| `llama3-70b-8192` | Groq | Llama 3 70B on Groq |
| `mixtral-8x7b-32768` | Groq | Mixtral MoE |
| `gemma2-9b-it` | Groq | Google Gemma 2 |
| `phi3:mini` | Ollama | Local — requires `ollama pull phi3:mini` |
| `llama3.2:3b` | Ollama | Local — requires `ollama pull llama3.2:3b` |
| `mistralai/Mistral-7B-Instruct-v0.2` | HuggingFace | Requires `HF_API_TOKEN` |

If the requested model is unavailable, the system automatically falls back to the next configured backend.

### 7. Flask API (`app.py`)

REST API with four endpoints, deployable via `python app.py` (dev) or Apache mod_wsgi (production):

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/chat` | POST | Main chat — accepts `message`, `model`, `conversation_id`, `top_k`, `max_tokens` |
| `/api/health` | GET | System status — LLM availability, index size, backend |
| `/api/info` | GET | Assistant metadata |
| `/api/rebuild-index` | POST | Rebuild RAG index after data changes |

---

## Anti-Hallucination Design

Four layers prevent the assistant from making things up:

1. **Explicit "no context" signal** — if nothing passes the similarity threshold, the LLM receives `[No relevant information found in the knowledge base for this query.]` and is instructed to say "I don't have that detail on hand" rather than inventing an answer from pre-trained knowledge.

2. **Strict system prompt** — explicitly instructs the LLM to answer only from the provided context. The fallback phrase ("I don't have that detail on hand") is tied specifically to the `[No relevant information found...]` marker — if any context was retrieved, the LLM must use it and answer rather than hedging. The model is also told which projects it is NOT (RAT 3.0, GRILSS, RECLAIM) to prevent self-confusion.

3. **Numbered, labelled context chunks with scores** — each retrieved chunk is formatted as `[N. source › section (score: 0.84)]` so the LLM can clearly attribute facts to specific sources and avoid conflating content from different projects.

4. **Large model (70B)** — `llama-3.3-70b-versatile` follows nuanced grounding instructions far more reliably than smaller 8B models, which tend to fill gaps with plausible-sounding but incorrect details.

---

## Why `BAAI/bge-base-en-v1.5` over `all-MiniLM-L6-v2`?

| Model | Dims | MTEB Score | Domain Retrieval | Size |
|-------|------|-----------|-----------------|------|
| `all-MiniLM-L6-v2` | 384 | ~56 | Good general | ~60MB |
| `BAAI/bge-base-en-v1.5` | 768 | ~63 | Significantly better on technical/domain text | ~90MB |

The higher-dimensional BGE model captures more nuanced semantic relationships — crucial for domain-specific queries like "what did Sanchit's RECLAIM framework achieve?" where the query and the chunk share little word overlap but high conceptual overlap.

---

## Why RAG instead of fine-tuning?

Fine-tuning bakes knowledge into model weights — updating it requires re-training. RAG keeps knowledge in editable files — updating it requires only running `build_index.py`. For a personal assistant where facts change (new projects, new publications, new opinions), RAG is far more practical.

Fine-tuning is still available via `scripts/fine_tune.py` for deeper style/personality customization, on top of the RAG layer.
