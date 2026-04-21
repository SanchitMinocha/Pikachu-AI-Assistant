import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Redirect HuggingFace/fastembed model cache into the project so www-data can access it.
# Fall back to ~/.cache/fastembed when the project models dir isn't writable by the
# current user (e.g. running build_index.py as a non-www-data developer account).
_project_models_dir = str(BASE_DIR / "data" / "models")
if os.access(_project_models_dir, os.W_OK):
    _models_dir = _project_models_dir
else:
    _models_dir = os.path.join(os.path.expanduser("~"), ".cache", "fastembed")
os.environ.setdefault("HF_HOME", _models_dir)
os.environ.setdefault("FASTEMBED_CACHE_PATH", _models_dir)

# Load .env file if present (works for direct `python app.py` and mod_wsgi)
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    with open(_env_file, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
WEBSITE_PAGES_DIR = DATA_DIR / "website_pages"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = PROCESSED_DIR / "vectorstore"
PERSONAL_DATA_PATH = DATA_DIR / "personal_data.json"
LOGS_DIR = BASE_DIR / "logs"

# --- LLM Backend ---
# Options: "ollama", "huggingface", "groq"
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# Ollama settings (local LLM server)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")  # lightweight default

# Cerebras Inference API (fast free fallback — cloud.cerebras.ai)
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama3.1-8b")

# OpenRouter API (aggregates many free models — openrouter.ai)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")

# HuggingFace Inference API (optional fallback — requires fine-grained token)
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")

# Groq API (fast free tier)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- RAG ---
TOP_K_RESULTS = 7
SIMILARITY_THRESHOLD = 0.3

# --- API ---
MAX_HISTORY_TURNS = 6   # number of conversation turns to keep
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.7
SECRET_KEY = os.getenv("SECRET_KEY", "sanchitai-secret-change-in-prod")
API_KEY = os.getenv("API_KEY", "")   # optional: protect /rebuild-index endpoint

# --- Identity (kept here for quick reference, full data in personal_data.json) ---
ASSISTANT_NAME = "Pikachu - Sanchit's AI Assistant"
CREATOR_NAME = "Sanchit Minocha"
