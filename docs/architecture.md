# How SanchitAI Works — Architecture

SanchitAI is a **RAG (Retrieval-Augmented Generation)** system. Instead of relying on a pre-trained model to memorize facts about Sanchit, it retrieves relevant information at query time and feeds it as context to an LLM. This makes it accurate, updatable, and grounded in real data.

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
      │         Embed query with Sentence-Transformers (ONNX)        │
      │         Search ChromaDB vector store (cosine similarity)     │
      │         Return top-K most relevant document chunks           │
      │                                                              │
      ├─── 2. AUGMENT ───────────────────────────────────────────────┤
      │         Inject retrieved chunks as CONTEXT into the prompt   │
      │         Append conversation history (last N turns)           │
      │         Apply SanchitAI system prompt (identity + rules)     │
      │                                                              │
      └─── 3. GENERATE ──────────────────────────────────────────────┘
                LLM generates grounded, human-like response
                Primary: Groq API (Llama 3.1 8B — fast, free)
                Fallback: Local Ollama (phi3:mini — slow, always available)
                         HuggingFace API (optional)
```

---

## Components

### 1. Knowledge Base (`data/knowledge_base/` + `data/personal_data.json`)

All facts about Sanchit live here as plain text:

| File | Contents |
|------|----------|
| `profile.md` | Education, career, skills, achievements |
| `projects.md` | Detailed project descriptions |
| `about_ai.md` | What SanchitAI is and how it was built |
| `github.md` | Auto-fetched GitHub repos (via `collect_web_data.py`) |
| `personal_data.json` | Editable personal data: hobbies, stories, opinions, FAQs |

### 2. Document Loader (`src/data/loader.py`)

Reads all knowledge files, splits them into ~500-character overlapping chunks, and prepares them for embedding. Chunking ensures that long documents fit within the LLM's context window while preserving sentence boundaries.

### 3. Embeddings (`src/rag/embeddings.py`)

Uses **fastembed** with `sentence-transformers/all-MiniLM-L6-v2` (ONNX, 384-dimensional vectors). Converts text chunks and user queries into semantic vector representations — words with similar meaning land close together in vector space.

### 4. Vector Store (`src/rag/vectorstore.py`)

**ChromaDB** stores the 300+ document embeddings on disk. At query time, cosine similarity search returns the top-K chunks most semantically related to the user's question in milliseconds.

### 5. LLM Assistant (`src/llm/assistant.py`)

Takes the retrieved context + conversation history and sends them to an LLM with a carefully crafted system prompt that gives the model its identity:
- Who it is (SanchitAI)
- Who built it (Sanchit Minocha)
- Its purpose (answer questions about Sanchit)
- Response style (concise, warm, human-like)

### 6. Flask API (`app.py`)

REST API with four endpoints, deployed via Apache mod_wsgi:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/chat` | POST | Main chat — accepts `message`, `model`, `conversation_id` |
| `/api/health` | GET | System status — LLM availability, index size |
| `/api/info` | GET | Assistant metadata |
| `/api/rebuild-index` | POST | Rebuild RAG index after data changes |

---

## Why RAG instead of fine-tuning?

Fine-tuning bakes knowledge into model weights — updating it requires re-training. RAG keeps knowledge in editable files — updating it requires only running `build_index.py`. For a personal assistant where facts change (new projects, new opinions), RAG is far more practical.

Fine-tuning is still available via `scripts/fine_tune.py` for deeper style/personality customization, on top of the RAG layer.
