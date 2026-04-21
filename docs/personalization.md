# Personalizing the Assistant

The assistant is designed to be fully customizable. All personal knowledge lives in editable files — no retraining needed for most updates.

---

## Quick Updates (No Retraining)

### Add hobbies, facts, opinions, stories

Edit `data/personal_data.json`. Look for the fields with `ADD_..._HERE` placeholders:

```json
{
  "hobbies": [
    "Poetry and creative writing",
    "Hiking",
    "Chess",
    "Playing guitar"
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
    "on_ai": "LLMs are only as good as the data behind them. RAG is what makes them actually useful for specific domains.",
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

FAQs are the most reliable way to ensure accurate, grounded answers to specific questions — the LLM will find them directly in the retrieved context.

After editing, rebuild the index:

```bash
sudo -u www-data bash -c "cd /var/www/html/scb && venv/bin/python scripts/build_index.py"
sudo systemctl restart apache2
```

---

## Add New Knowledge Files

Drop any `.md` or `.pdf` file into `data/knowledge_base/` — it will be automatically indexed on the next build. Use this for:

- Publications list (`publications.md`)
- Interview write-ups
- Blog post summaries
- Detailed project write-ups
- CV / resume (`Sanchit_CV.pdf` is already included)

Example `data/knowledge_base/publications.md`:

```markdown
# Sanchit Minocha – Publications

## 2024

### RAT 3.0: A Scalable Satellite-based Reservoir Monitoring Tool
- **Journal:** Journal of Hydrology
- **Summary:** Describes the architecture and validation of RAT 3.0...

## 2023
...
```

---

## Keeping Answers Accurate and Grounded

The assistant is configured to answer **only from retrieved context** — it will not hallucinate or fill gaps with invented facts. To keep answers accurate:

- **For specific facts** (project names, dates, publications): add them as FAQ entries in `personal_data.json`. FAQ entries are loaded as individual high-priority documents and are always retrieved first for matching questions.
- **For broad topics** (career, skills, personality): keep `profile.md` and `projects.md` up to date.
- **For the CV**: replace `data/knowledge_base/Sanchit_CV.pdf` with an updated version and rebuild the index.
- **For follow-up questions**: the retriever automatically uses recent conversation history to find the right context — no special handling needed.

After any update, always rebuild:

```bash
python scripts/build_index.py
```

---

## Refresh GitHub Data

To pull the latest repos from GitHub:

```bash
source venv/bin/activate
python scripts/collect_web_data.py
sudo -u www-data bash -c "cd /var/www/html/scb && venv/bin/python scripts/build_index.py"
```

This updates `data/knowledge_base/github.md` with your latest repositories.

---

## Optional: Fine-Tune the Model

For deeper personality/style customization beyond RAG, you can LoRA fine-tune a small model on your personal Q&A data. Note: this is optional — RAG alone already provides accurate, grounded answers.

### Step 1 — Generate training data

```bash
source venv/bin/activate
python scripts/fine_tune.py --data-only
```

This creates `models/sanchitai-ft/training_data/train.jsonl` from your `personal_data.json`. Inspect it to make sure the Q&A pairs look right.

### Step 2 — Run fine-tuning (GPU recommended)

```bash
# Install fine-tuning dependencies first
pip install unsloth transformers datasets peft trl accelerate bitsandbytes

# Fine-tune phi3:mini (fastest, good quality)
python scripts/fine_tune.py --model phi3 --output ./models/sanchitai-ft

# Or fine-tune Llama 3.2 3B (better quality)
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

The fine-tuned model will now be used as the local fallback (or primary if `LLM_BACKEND=ollama`).

---

## Adapting for a Different Person

This project can be forked and used as a personal AI assistant template for anyone. To adapt it:

1. Replace all content in `data/knowledge_base/` with your own profile
2. Replace `data/personal_data.json` with your own data
3. Update `ASSISTANT_NAME` and `CREATOR_NAME` in `config.py`
4. Update the system prompt in `src/llm/assistant.py` — especially the identity section and the list of projects the assistant is NOT
5. Rebuild the index: `python scripts/build_index.py`
