# Personalizing the Assistant

The assistant is designed to be fully customizable. All personal knowledge lives in editable files — no retraining needed for most updates.

---

## Quick Updates (No Retraining)

### Add hobbies, facts, opinions, stories

Edit `data/personal_data.json`. Look for fields with `ADD_..._HERE` placeholders:

```json
{
  "hobbies": [
    "Poetry and creative writing",
    "Hiking"
  ],
  "personal_facts": [
    "Born and raised in Chandigarh, India",
    "Favourite book is 'The Alchemist' by Paulo Coelho"
  ],
  "personal_stories": [
    {
      "title": "Why I pivoted to water research",
      "story": "One afternoon in 2021, I was looking at satellite images of shrinking reservoirs..."
    }
  ],
  "opinions": {
    "on_ai": "LLMs are only as good as the data behind them.",
    "on_climate": "Water scarcity is the most underrated consequence of climate change."
  },
  "frequently_asked_questions": [
    {
      "question": "What is Sanchit's favourite programming language?",
      "answer": "Python, without question."
    }
  ]
}
```

FAQs are the most reliable way to ensure accurate answers to specific questions — each FAQ entry becomes its own indexed document and is retrieved directly when someone asks a matching question.

After editing, rebuild the index:

```bash
source venv/bin/activate
python scripts/build_index.py
```

---

## Update Publications

Publications live in `data/website_pages/publications.json`. Each entry is automatically indexed as a self-contained chunk (title + authors + journal + year + abstract) — so queries like *"what papers did Sanchit publish in 2025?"* retrieve the exact publication entries.

Format for a new entry:

```json
{
  "title": "Your Paper Title",
  "authors": "<strong>Sanchit Minocha</strong>, Co-Author Name",
  "journal": "Journal Name, Volume(Issue), Pages, Year.",
  "doi": "https://doi.org/...",
  "pdf": "",
  "abstract": "Full abstract text here — this is what the model retrieves.",
  "year": "2025"
}
```

After editing, rebuild the index. No other changes needed.

---

## Update Portfolio Projects

Portfolio projects live in `data/website_pages/portfolios.json`. Each entry is automatically indexed. The following fields are included in retrieval context:

- `title`, `meta`, `tagline`, `impact`, `description`
- `details.techStack`
- `details.overviewBase`
- `details.challenge`
- `details.solution`

Bare URLs are excluded from chunks — they don't help retrieval and cause degenerate chunking.

---

## Add New Knowledge Files

Drop any `.md` or `.pdf` file into `data/knowledge_base/` — it will be automatically indexed on the next build.

**Markdown files** are split by `##` headers — each section becomes one chunk. Structure your file with clear `##` headings:

```markdown
# Sanchit Minocha – Additional Notes

## Teaching Experience

Sanchit has been a Teaching Assistant for CEWA 565 (Remote Sensing for Water Resources) at UW...

## Awards

- IIT Roorkee Gold Medal (2019)
- AGU Outstanding Student Presentation Award (2023)
```

**PDF files** are chunked using a sliding window — works well for CVs and papers.

Use this for:
- Additional project write-ups
- Teaching notes
- Interview write-ups
- Any long-form biographical content

---

## Keeping Answers Accurate and Grounded

The assistant answers **only from retrieved context** — it never hallucinate or fills gaps with invented facts. To keep answers accurate:

- **For specific facts** (project names, dates, publications): add them as FAQ entries in `personal_data.json`. FAQ entries are loaded as individual high-priority documents.
- **For broad topics** (career, skills, personality): keep `profile.md` and `projects.md` up to date, using `##` headers to organize sections.
- **For publications**: keep `data/website_pages/publications.json` updated — each entry is one chunk, so full abstract text is indexed and retrievable.
- **For the CV**: replace `data/knowledge_base/Sanchit_CV.pdf` with an updated version and rebuild.
- **For follow-up questions**: the retriever automatically uses recent conversation history to find the right context — no special handling needed.

After any update, rebuild:

```bash
python scripts/build_index.py
```

---

## Refresh GitHub Data

To pull the latest repos from GitHub:

```bash
source venv/bin/activate
python scripts/collect_web_data.py
python scripts/build_index.py
```

This updates `data/knowledge_base/github.md` with your latest repositories and re-indexes everything.

---

## Debug Retrieval Issues

If the assistant is giving wrong or vague answers, check what chunks are actually being retrieved:

```bash
# All retrieval is logged at INFO level — watch live
tail -f logs/app.log | grep -E 'score=|No chunks'
```

Each line shows the similarity score, source file, section, and first 120 characters of the chunk. If the expected chunk isn't appearing in the top-7, the answer will be wrong no matter how good the LLM is.

Common fixes:
- **Low scores for specific topics:** Add an FAQ entry in `personal_data.json` — FAQ entries are very reliably retrieved for exact question matches.
- **Publications not appearing:** Make sure `data/website_pages/publications.json` has complete abstract text (not just title + journal).
- **Old data persisting:** Run `python scripts/build_index.py` to rebuild — the `--no-clear` flag appends without wiping.

---

## Optional: Fine-Tune the Model

For deeper personality/style customization beyond RAG, you can LoRA fine-tune a small model on your personal Q&A data. RAG alone already provides accurate, grounded answers — fine-tuning is purely for style.

### Step 1 — Generate training data

```bash
source venv/bin/activate
python scripts/fine_tune.py --data-only
```

This creates `models/sanchitai-ft/training_data/train.jsonl` from your `personal_data.json`.

### Step 2 — Run fine-tuning (GPU recommended)

```bash
pip install unsloth transformers datasets peft trl accelerate bitsandbytes

python scripts/fine_tune.py --model phi3 --output ./models/sanchitai-ft
# Or for better quality:
python scripts/fine_tune.py --model llama3.2 --output ./models/sanchitai-ft
```

### Step 3 — Load into Ollama

```bash
ollama create sanchitai -f ./models/sanchitai-ft/Modelfile
```

### Step 4 — Update config

In `.env`:
```
OLLAMA_MODEL=sanchitai
```

---

## Adapting for a Different Person

This project can be forked and used as a personal AI assistant template for anyone. To adapt it:

1. Replace all content in `data/knowledge_base/` with your own profile
2. Replace `data/personal_data.json` with your own data
3. Replace `data/website_pages/publications.json` and `portfolios.json` with your own
4. Update `ASSISTANT_NAME` and `CREATOR_NAME` in `config.py`
5. Update the system prompt in `src/llm/assistant.py` — especially the identity section and the list of projects the assistant is NOT
6. Rebuild the index: `python scripts/build_index.py`
