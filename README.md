# SanchitAI – Personal AI Assistant

> A RAG + LLM personal chatbot API deployed on Apache Server.
> Built by **[Sanchit Minocha](https://sanchitminocha.github.io/)** as an end-to-end NLP engineering project.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)](https://trychroma.com)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20%2B%20Groq-purple)](https://ollama.com)
[![Apache](https://img.shields.io/badge/Server-Apache%20mod__wsgi-red)](https://httpd.apache.org)

---

## What is SanchitAI?

SanchitAI is a personal AI assistant that answers questions about Sanchit Minocha — his research, career, projects, opinions, and personality. It is deployed as a REST API and integrated as a chatbot on his [personal website](https://sanchitminocha.github.io/).

Ask it things like:
- *"What is Sanchit's most outstanding work?"*
- *"What does Sanchit think about LLMs and RAG?"*
- *"Tell me about RAT 3.0."*
- *"What are his hobbies?"*

---

## How It Works

SanchitAI uses **Retrieval-Augmented Generation (RAG)**. When you ask a question:

1. Your question is converted to a semantic vector (fastembed, ONNX)
2. ChromaDB finds the most relevant chunks from the knowledge base
3. Those chunks are injected as context into an LLM prompt
4. The LLM generates a grounded, human-like response

**LLM backends** (with automatic fallback):
- **Primary:** Groq API — Llama 3.1 8B (fast, free cloud)
- **Fallback:** Local Ollama — phi3:mini (always available, no internet needed)

For a deeper technical explanation → [docs/architecture.md](docs/architecture.md)

---

## API

### `POST /api/chat`

```bash
curl -X POST https://yourserver.com/scb/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Who is Sanchit Minocha?",
    "model": "phi3:mini",          // optional — overrides default, falls back if unavailable
    "conversation_id": "abc-123",  // optional — enables multi-turn conversation
    "top_k": 5                     // optional — number of RAG results (default 5)
  }'
```

**Response:**
```json
{
  "response": "Sanchit Minocha is a PhD researcher at the University of Washington...",
  "model": "llama-3.1-8b-instant",
  "sources": ["profile.md", "personal_data.json"],
  "conversation_id": "abc-123",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | LLM status, index size, backend availability |
| `/api/info` | GET | Assistant metadata and creator info |
| `/api/rebuild-index` | POST | Rebuild RAG index after editing knowledge files |

---

## Quick Start

```bash
git clone https://github.com/SanchitMinocha/sanchitai.git
cd sanchitai

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # fill in your GROQ_API_KEY
python scripts/build_index.py
python app.py               # dev server on :5050
```

For full Apache deployment → [docs/deployment.md](docs/deployment.md)

---

## Personalization

All knowledge about Sanchit lives in plain editable files:

```
data/
├── personal_data.json        ← hobbies, stories, opinions, FAQs — edit freely
└── knowledge_base/
    ├── profile.md            ← career, education, skills
    ├── projects.md           ← project details
    └── about_ai.md           ← what SanchitAI is
```

After editing any file, rebuild the index:

```bash
python scripts/build_index.py
```

For adding publications, fine-tuning the model, or adapting for a different person → [docs/personalization.md](docs/personalization.md)

---

## Security — What NOT to Commit

| File | Why | What to commit instead |
|------|-----|----------------------|
| `.env` | Contains API keys (Groq, HuggingFace) | `.env.example` (placeholder keys) |
| `data/processed/` | Generated ChromaDB vector store | Rebuilt by `build_index.py` |
| `data/models/` | Downloaded ONNX model files (~90MB) | Rebuilt on first run |
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
│
├── data/
│   ├── personal_data.json    ← EDIT THIS to personalize
│   └── knowledge_base/       Markdown knowledge files
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
