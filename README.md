# Pikachu – Sanchit's AI Assistant

> A RAG + LLM personal chatbot API deployed on Apache Server.
> Built by **[Sanchit Minocha](https://sanchitminocha.github.io/)** as an end-to-end NLP engineering project.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)](https://trychroma.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%2B%20Ollama-purple)](https://groq.com)
[![Apache](https://img.shields.io/badge/Server-Apache%20mod__wsgi-red)](https://httpd.apache.org)

---

## What is Pikachu?

Pikachu is Sanchit's personal AI assistant — it answers questions about Sanchit Minocha: his research, career, projects, publications, opinions, and personality. It is deployed as a REST API and integrated as a chatbot on his [personal website](https://sanchitminocha.github.io/).

Ask it things like:
- *"What is Sanchit's most impactful project?"*
- *"Tell me about the GRILSS dataset."*
- *"What publications has Sanchit written in 2025?"*
- *"What are his hobbies?"*

---

## How It Works

Pikachu uses **Retrieval-Augmented Generation (RAG)**. When you ask a question:

1. The query is cleaned — person's name and question scaffolding are stripped so the embedding focuses on the actual topic
2. Follow-up detection: if the query contains a dangling pronoun ("it", "that", "those") or is a continuation phrase ("yes", "tell me more", "other than that?"), the last user message *and* a snippet of the last assistant response are prepended — this grounds the embedding in prior context while steering retrieval toward *new* content not yet covered
3. The augmented query is embedded with **BAAI/bge-base-en-v1.5** (ONNX, 768-dim) and ChromaDB finds the most relevant chunks via cosine similarity
4. Chunks scoring below the similarity threshold are discarded — the LLM receives an explicit "no information found" marker rather than empty context
5. Surviving chunks are injected as numbered, labelled context into the LLM prompt
6. The LLM generates a response grounded strictly in that context; the fallback "I don't know" phrase only fires when the context block is truly empty

**LLM backends** (with automatic fallback in order):
1. **Groq** — `llama-3.3-70b-versatile` (fast, free tier)
2. **Cerebras** — `llama3.3-70b` (fast, free — `CEREBRAS_API_KEY`)
3. **OpenRouter** — `openai/gpt-oss-20b:free` (free models — `OPENROUTER_API_KEY`)
4. **HuggingFace** — `meta-llama/Llama-3.2-3B-Instruct` (requires fine-grained token)
5. **Ollama** — local, no internet needed

For a deeper technical explanation → [docs/architecture.md](docs/architecture.md)

---

## API

### `POST /api/chat`

```bash
curl -X POST http://localhost:5050/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Who is Sanchit Minocha?",
    "conversation_id": "abc-123",
    "top_k": 7,
    "max_tokens": 500,
    "model": "llama-3.3-70b-versatile"
  }'
```

**Request fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | string | required | The question to ask |
| `conversation_id` | string | auto-generated | Pass back the returned ID for multi-turn conversation |
| `top_k` | int | 7 | Number of RAG chunks to retrieve (max 10) |
| `max_tokens` | int | 500 | Maximum tokens in the response |
| `model` | string | from `.env` | Override the LLM (see accepted values below) |

**Accepted `model` values:**

| Value | Backend | Notes |
|-------|---------|-------|
| `llama-3.3-70b-versatile` | Groq | Best quality — recommended default |
| `llama-3.1-70b-versatile` | Groq | Good quality |
| `llama-3.1-8b-instant` | Groq | Fast, lower accuracy |
| `llama3-8b-8192` | Groq | Groq-hosted Llama 3 8B |
| `llama3-70b-8192` | Groq | Groq-hosted Llama 3 70B |
| `mixtral-8x7b-32768` | Groq | Mixtral MoE |
| `gemma2-9b-it` | Groq | Google Gemma 2 9B |
| `gemma-7b-it` | Groq | Google Gemma 7B |
| `llama3.3-70b` | Cerebras | Free — set `CEREBRAS_API_KEY` in `.env` |
| `llama3.1-70b` | Cerebras | Free — fast 70B model |
| `llama3.1-8b` | Cerebras | Free — fast, smaller model |
| `gpt-oss-120b` | Cerebras | Free — large OSS model |
| `qwen-3-235b-a22b-instruct-2507` | Cerebras | Free — Qwen 3 235B MoE |
| `openai/gpt-oss-20b:free` | OpenRouter | Free — default; set `OPENROUTER_API_KEY` in `.env` |
| `meta-llama/llama-3.2-3b-instruct:free` | OpenRouter | Free — lightweight Llama |
| `google/gemma-3-1b-it:free` | OpenRouter | Free — lightweight Gemma |
| `phi3:mini` | Ollama (local) | Requires `ollama pull phi3:mini` |
| `llama3.2:3b` | Ollama (local) | Requires `ollama pull llama3.2:3b` |
| `meta-llama/Llama-3.2-3B-Instruct` | HuggingFace | Requires fine-grained token with inference permission |

**Response:**
```json
{
  "response": "Sanchit Minocha is a PhD researcher at the University of Washington...",
  "model": "llama-3.3-70b-versatile",
  "sources": ["profile.md", "personal_data.json"],
  "conversation_id": "abc-123",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /api/health` | GET | LLM status, index size, backend availability |
| `GET /api/info` | GET | Assistant metadata and creator info |
| `POST /api/rebuild-index` | POST | Rebuild RAG index after editing knowledge files |

---

## Quick Start

```bash
git clone https://github.com/SanchitMinocha/Pikachu-AI-Assistant.git
cd sanchitai

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # fill in your GROQ_API_KEY
python scripts/build_index.py
python app.py               # dev server on :5050
```

For full Apache deployment → [docs/deployment.md](docs/deployment.md)

---

## Testing

### Browser (easiest)

Open `test.html` in any browser — it provides a full chat UI with model selector, source display, and health status. Works against the dev server or Apache:

```
# Dev server
open test.html        # macOS
xdg-open test.html    # Linux

# Or just drag-and-drop into your browser
```

The default base URL in the tester is `http://localhost:5050`. You can change it in the Settings panel at the top of the page.

### Terminal — quick checks

The server uses a self-signed SSL certificate, so pass `-k` to skip certificate verification:

```bash
# Health check
curl -sk https://f-hossain-3.ce.washington.edu/scb/api/health | python3 -m json.tool

# Simple chat
curl -sk -X POST https://f-hossain-3.ce.washington.edu/scb/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about RAT 3.0"}' | python3 -m json.tool

# Multi-turn conversation (reuse conversation_id)
curl -sk -X POST https://f-hossain-3.ce.washington.edu/scb/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What about GRILSS?", "conversation_id": "test-1"}' \
  | python3 -m json.tool

# Try a specific model
curl -sk -X POST https://f-hossain-3.ce.washington.edu/scb/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What publications did Sanchit write in 2025?", "model": "llama-3.3-70b-versatile"}' \
  | python3 -m json.tool

# Rebuild the index after editing knowledge files
curl -sk -X POST https://f-hossain-3.ce.washington.edu/scb/api/rebuild-index | python3 -m json.tool

# Assistant info
curl -sk https://f-hossain-3.ce.washington.edu/scb/api/info | python3 -m json.tool
```

For local dev server (`python app.py`), drop the `-k` and use `http://localhost:5050`.

---

## Knowledge Base

All knowledge about Sanchit lives in plain editable files:

```
data/
├── personal_data.json          ← hobbies, stories, opinions, FAQs — edit freely
├── knowledge_base/
│   ├── profile.md              ← career, education, skills
│   ├── projects.md             ← project details (split by ## section)
│   ├── about_ai.md             ← what this assistant is
│   ├── github.md               ← auto-fetched GitHub repos
│   └── Sanchit_CV.pdf          ← CV (auto-indexed as PDF)
└── website_pages/
    ├── publications.json       ← research publications (auto-indexed)
    └── portfolios.json         ← portfolio projects (auto-indexed)
```

Each `##`-headed section in Markdown files becomes one chunk. Publications and portfolio entries each become their own self-contained chunks — so asking about a specific paper retrieves exactly that paper's context.

After editing any file, rebuild the index:

```bash
python scripts/build_index.py
```

For adding more data, managing publications, or adapting for a different person → [docs/personalization.md](docs/personalization.md)

---

## Security — What NOT to Commit

| File | Why | What to commit instead |
|------|-----|----------------------|
| `.env` | Contains API keys (Groq, HuggingFace) | `.env.example` (placeholder keys) |
| `data/processed/` | Generated ChromaDB vector store | Rebuilt by `build_index.py` |
| `data/models/` | Downloaded ONNX model files (~90MB) | Auto-downloaded on first run |
| `models/` | Fine-tuned model weights (GBs) | — |
| `logs/` | Server logs | — |

All of the above are already in `.gitignore`.

---

## Project Structure

```
sanchitai/
├── app.py                    API entry point (Flask)
├── config.py                 All settings
├── wsgi.py                   Apache mod_wsgi entry point
├── requirements.txt
├── .env.example              Safe config template (commit this)
├── test.html                 Browser-based chat tester
│
├── data/
│   ├── personal_data.json    ← EDIT THIS to personalize
│   ├── knowledge_base/       Markdown + PDF knowledge files
│   └── website_pages/        ← publications.json + portfolios.json
│
├── src/
│   ├── rag/                  Embeddings + ChromaDB + retriever
│   ├── llm/                  Groq / Ollama / HuggingFace backends
│   └── data/                 Document loading and chunking
│
├── scripts/
│   ├── build_index.py        Build RAG vector index
│   ├── collect_web_data.py   Refresh GitHub data
│   └── fine_tune.py          Optional LoRA fine-tuning
│
├── docs/
│   ├── architecture.md       How it works (RAG pipeline explained)
│   ├── deployment.md         Apache deployment step-by-step
│   └── personalization.md    How to update data and retrain
│
└── apache/
    └── sanchit-ai.conf       Apache VirtualHost config
```

---

## About the Creator

**Sanchit Minocha** is a PhD researcher at the University of Washington (Geospatial AI & Water Resource Management), IIT Roorkee Gold Medalist, and data scientist with 7+ years of experience building tools that use satellite data to manage global water resources.

- Website: [sanchitminocha.github.io](https://sanchitminocha.github.io/)
- GitHub: [github.com/SanchitMinocha](https://github.com/SanchitMinocha)
- LinkedIn: [linkedin.com/in/sanchitminochaiitr](https://www.linkedin.com/in/sanchitminochaiitr/)

---

*MIT License — fork and adapt for your own personal AI assistant.*
