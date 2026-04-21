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
      │         Embed query with Sentence-Transformers (ONNX)        │
      │         Search ChromaDB vector store (cosine similarity)     │
      │         Return top-K most relevant document chunks           │
      │                                                              │
      ├─── 2. AUGMENT ───────────────────────────────────────────────┤
      │         Format chunks as numbered, labelled context blocks   │
      │         Append conversation history (last N turns)           │
      │         Apply system prompt (identity + strict grounding)    │
      │                                                              │
      └─── 3. GENERATE ──────────────────────────────────────────────┘
                LLM generates a response grounded ONLY in context
                Primary: Groq API (Llama 3.3 70B — fast, accurate)
                Fallback 1: Local Ollama (phi3:mini — offline capable)
                Fallback 2: HuggingFace API (optional)
```

---

## Components

### 1. Knowledge Base (`data/knowledge_base/` + `data/personal_data.json`)

All facts about Sanchit live here as plain text:

| File | Contents |
|------|----------|
| `profile.md` | Education, career, skills, achievements |
| `projects.md` | Detailed project descriptions |
| `about_ai.md` | What this assistant is and how it was built |
| `github.md` | Auto-fetched GitHub repos (via `collect_web_data.py`) |
| `Sanchit_CV.pdf` | Full CV — auto-extracted and indexed as text chunks |
| `personal_data.json` | Editable personal data: hobbies, stories, opinions, FAQs |

### 2. Document Loader (`src/data/loader.py`)

Reads all knowledge files — Markdown, PDF, and JSON — splits them into ~500-character overlapping chunks, and prepares them for embedding. PDF text is extracted via `pypdf`. Chunking preserves sentence boundaries and ensures long documents fit within the LLM's context window.

### 3. Embeddings (`src/rag/embeddings.py`)

Uses **fastembed** with `sentence-transformers/all-MiniLM-L6-v2` (ONNX, 384-dimensional vectors). Converts text chunks and user queries into semantic vector representations — words with similar meaning land close together in vector space.

### 4. Vector Store (`src/rag/vectorstore.py`)

**ChromaDB** stores the 390+ document embeddings on disk. At query time, cosine similarity search returns the top-K chunks most semantically related to the user's question in milliseconds.

### 5. Retriever (`src/rag/retriever.py`)

For follow-up questions ("what is the name of that tool?"), the retriever augments the query with the last few conversation turns before embedding — so context-dependent questions still retrieve the right chunks. Retrieved chunks are returned as numbered, labelled blocks (e.g. `[1. projects.md › projects]`) so the LLM can clearly distinguish between separate projects and sources.

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

REST API with four endpoints, deployed via Apache mod_wsgi:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/chat` | POST | Main chat — accepts `message`, `model`, `conversation_id`, `top_k`, `max_tokens` |
| `/api/health` | GET | System status — LLM availability, index size |
| `/api/info` | GET | Assistant metadata |
| `/api/rebuild-index` | POST | Rebuild RAG index after data changes |

---

## Anti-Hallucination Design

Three layers prevent the assistant from making things up:

1. **Strict system prompt** — explicitly instructs the LLM to answer only from the provided context and say "I don't have that detail" when context is insufficient. The model is told which projects it is NOT (RAT 3.0, GRILSS, RECLAIM) to prevent self-confusion.

2. **Numbered, labelled context chunks** — each retrieved chunk is formatted as `[N. source › section]` so the LLM can clearly attribute facts to specific sources and avoid conflating content from different projects.

3. **Larger model (70B)** — `llama-3.3-70b-versatile` follows nuanced grounding instructions far more reliably than smaller 8B models, which tend to paraphrase and fill gaps with plausible-sounding but incorrect details.

---

## Why RAG instead of fine-tuning?

Fine-tuning bakes knowledge into model weights — updating it requires re-training. RAG keeps knowledge in editable files — updating it requires only running `build_index.py`. For a personal assistant where facts change (new projects, new publications, new opinions), RAG is far more practical.

Fine-tuning is still available via `scripts/fine_tune.py` for deeper style/personality customization, on top of the RAG layer.
