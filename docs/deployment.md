# Deployment Guide

## Prerequisites

- Ubuntu 20.04+ server
- Apache 2.4 with `mod_wsgi` enabled
- Python 3.10+
- 4GB+ RAM (8GB recommended if running local Ollama)

---

## 1. Clone and Install

```bash
git clone https://github.com/SanchitMinocha/sanchitai.git
cd sanchitai

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys. The minimum required for cloud mode:

```
LLM_BACKEND=groq
GROQ_API_KEY=<your_groq_key>     # free at console.groq.com
SECRET_KEY=<random_string>
```

For local-only mode (no internet LLM needed):
```
LLM_BACKEND=ollama
OLLAMA_MODEL=phi3:mini
```

---

## 3. Install and Start Ollama (local fallback)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini            # ~2.2GB download
sudo systemctl enable ollama
sudo systemctl start ollama
```

---

## 4. Build the RAG Index

```bash
# Fetch latest GitHub data (optional)
python scripts/collect_web_data.py

# Build vector index — run as www-data so Apache can read it
sudo -u www-data bash -c "
  cd $(pwd) &&
  HF_HOME=$(pwd)/data/models \
  FASTEMBED_CACHE_PATH=$(pwd)/data/models \
  venv/bin/python scripts/build_index.py
"
```

---

## 5. Apache Configuration

Add the following block inside your existing `<VirtualHost>` in `/etc/apache2/sites-available/your-site.conf`:

```apache
# SanchitAI Flask API
WSGIDaemonProcess sanchitai \
    python-home=/var/www/html/scb/venv \
    python-path=/var/www/html/scb \
    processes=2 threads=4

WSGIScriptAlias /scb /var/www/html/scb/wsgi.py \
    process-group=sanchitai

<Directory /var/www/html/scb>
    <Files wsgi.py>
        Require all granted
    </Files>
</Directory>
```

Then:

```bash
sudo apache2ctl configtest        # verify no syntax errors
sudo systemctl reload apache2
```

The API is now live at `https://yourserver.com/scb/api/`.

---

## 6. Verify Deployment

```bash
# Health check
curl -k https://yourserver.com/scb/api/health | python3 -m json.tool

# Test chat
curl -k -X POST https://yourserver.com/scb/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Who is Sanchit Minocha?"}' | python3 -m json.tool
```

A healthy response looks like:
```json
{
  "response": "Sanchit Minocha is a PhD researcher...",
  "model": "llama-3.1-8b-instant",
  "sources": ["profile.md", "personal_data.json"],
  "conversation_id": "..."
}
```

---

## 7. Permissions Checklist

The `www-data` user (Apache) needs write access to:

```bash
sudo chown -R www-data:www-data /var/www/html/scb/data/processed   # ChromaDB
sudo chown -R www-data:www-data /var/www/html/scb/data/models       # fastembed cache
sudo chown -R www-data:www-data /var/www/html/scb/logs              # log files
```

---

## 8. Testing a Specific Model

Pass `"model"` in the request body to override the default:

```bash
# Use local Ollama model
curl -k -X POST https://yourserver.com/scb/api/chat \
  -d '{"message": "Who are you?", "model": "phi3:mini"}' \
  -H "Content-Type: application/json"

# Use a larger Groq model
curl -k -X POST https://yourserver.com/scb/api/chat \
  -d '{"message": "Who are you?", "model": "llama-3.1-70b-versatile"}' \
  -H "Content-Type: application/json"
```

If the requested model fails, the system automatically falls back to the next available backend.

---

## Updating the Index After Data Changes

Any time you edit `data/personal_data.json` or knowledge base files, rebuild:

```bash
sudo -u www-data bash -c "cd /var/www/html/scb && venv/bin/python scripts/build_index.py"
```

Or hit the API endpoint:

```bash
curl -k -X POST https://yourserver.com/scb/api/rebuild-index \
  -H "X-API-Key: your_api_key"    # only needed if API_KEY is set in .env
```
