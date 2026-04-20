# About SanchitAI

## What is SanchitAI?

SanchitAI is a personal AI assistant built by **Sanchit Minocha**, a PhD researcher at the University of Washington. It is designed to serve as Sanchit's digital representative — answering questions about his career, research, projects, skills, opinions, and personality in a helpful, human-like way.

SanchitAI was built as a **NLP + RAG + LLM project**, demonstrating Sanchit's engineering skills across the full stack of modern AI systems.

---

## Who Built SanchitAI?

**Sanchit Minocha** built SanchitAI entirely himself. He designed the architecture, curated the training data, implemented the RAG pipeline, integrated the LLM, and deployed it on Apache Server.

Sanchit chose to build this as a showcase project because:
1. It demonstrates practical NLP engineering (not just using an API)
2. It gives him a persistent, personalized digital presence
3. It's a fun and useful project that integrates with his personal website

---

## How Does SanchitAI Work?

SanchitAI uses a **RAG (Retrieval-Augmented Generation)** architecture:

1. **Knowledge Base:** Sanchit's profile, career, projects, opinions, and personal data are stored as documents (markdown + JSON).
2. **Vector Embeddings:** Documents are chunked and encoded into semantic vector embeddings using Sentence-Transformers (`all-MiniLM-L6-v2`).
3. **Vector Store:** Embeddings are stored in ChromaDB for fast semantic retrieval.
4. **Retrieval:** When a question is asked, the system retrieves the most relevant document chunks.
5. **Generation:** The retrieved context + conversation history is sent to an LLM (Llama 3.2 via Ollama) to generate a grounded, human-like response.

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| API Framework | Python Flask |
| LLM Backend | Ollama (Llama 3.2:3b) with HuggingFace/Groq fallback |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (persistent) |
| Web Server | Apache (mod_wsgi) |
| Data Format | Markdown + JSON knowledge base |
| Language | Python 3.10+ |

---

## What Can SanchitAI Answer?

SanchitAI can answer questions about:
- Sanchit's education and academic background
- His research and publications
- His professional experience and projects
- His technical skills and expertise
- His opinions on technology, research, and the environment
- His personality, interests, and hobbies
- His philosophy and goals
- General questions about his AI tools (RAT 3.0, GRILSS, RECLAIM)
- Why he made certain career choices
- What his most outstanding work is

---

## Limitations

SanchitAI is designed to answer questions about Sanchit. For questions outside this domain, it may redirect or note that the question is beyond its scope. It is not a general-purpose AI — it is Sanchit's personal assistant.

---

## API Integration

SanchitAI is deployed as a REST API and is integrated into Sanchit's personal website (https://sanchitminocha.github.io/) as a chat widget. The API is open for use to power the chatbot experience.

**API Endpoint:** `POST /api/chat`

```json
{
  "message": "What is Sanchit's most outstanding work?",
  "conversation_id": "optional-session-id"
}
```

---

## Project Repository

The full source code for SanchitAI is open-sourced on GitHub as part of Sanchit's portfolio of NLP and AI projects.
