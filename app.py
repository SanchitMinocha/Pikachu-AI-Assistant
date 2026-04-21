"""
SanchitAI – Personal AI Assistant API
Built by Sanchit Minocha

Endpoints:
  POST /api/chat           - Chat with SanchitAI
  GET  /api/health         - Health check
  GET  /api/info           - Assistant info
  POST /api/rebuild-index  - Rebuild the RAG vector index (requires API_KEY if set)
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Bootstrap path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.rag.retriever import retrieve, format_context, reload_vectorstore, build_retrieval_query
from src.llm.assistant import generate, check_ollama_available, list_ollama_models

# ----- Logging -----
config.LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOGS_DIR / "app.log"),
    ],
)
logger = logging.getLogger(__name__)

# ----- Flask App -----
app = Flask(__name__)
app.secret_key = config.SECRET_KEY
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Allow all origins for chatbot integration

# In-memory conversation history (per session)
# Format: { conversation_id: [{"role": "user"|"assistant", "content": "..."}] }
_conversations: dict = {}


def get_or_create_conversation(conversation_id: str = None):
    if not conversation_id or conversation_id not in _conversations:
        conversation_id = conversation_id or str(uuid.uuid4())
        _conversations[conversation_id] = []
    # Trim old conversations (keep max 100 sessions in memory)
    if len(_conversations) > 100:
        oldest = next(iter(_conversations))
        del _conversations[oldest]
    return conversation_id, _conversations[conversation_id]


def require_api_key(f):
    """Optional API key protection for sensitive endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if config.API_KEY:
            key = request.headers.get("X-API-Key") or request.args.get("api_key")
            if key != config.API_KEY:
                return jsonify({"error": "Unauthorized. Provide a valid API key."}), 401
        return f(*args, **kwargs)
    return decorated


# ----- Routes -----

@app.route("/test")
def test_page():
    return send_from_directory(str(Path(__file__).parent), "test.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.

    Request JSON:
      {
        "message": "What is Sanchit's most outstanding work?",
        "conversation_id": "optional-uuid",
        "top_k": 5,             (optional, number of RAG results)
        "model": "phi3:mini"    (optional, override model — falls back if unavailable)
      }

    Response JSON:
      {
        "response": "...",
        "conversation_id": "uuid",
        "sources": ["profile.md", ...],
        "model": "llama3.2:3b",
        "timestamp": "..."
      }
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400

    if len(message) > 2000:
        return jsonify({"error": "Message too long (max 2000 characters)"}), 400

    conversation_id = data.get("conversation_id")
    top_k = min(int(data.get("top_k", config.TOP_K_RESULTS)), 10)
    max_tokens = int(data.get("max_tokens", config.MAX_NEW_TOKENS))
    model_override = (data.get("model") or "").strip() or None

    # Get or create conversation history
    conv_id, history = get_or_create_conversation(conversation_id)

    # RAG: build topic-focused retrieval query (strips name + question scaffolding)
    retrieval_query = build_retrieval_query(message, history)
    retrieved = retrieve(retrieval_query, top_k=top_k)
    context = format_context(retrieved)
    sources = list({doc["metadata"].get("source", "") for doc in retrieved})

    logger.info(
        f"[{conv_id[:8]}] Query: '{message[:80]}' | "
        f"Retrieved {len(retrieved)} chunks from {sources}"
    )

    # Generate response
    try:
        response_text, model_used = generate(
            user_message=message,
            context=context,
            history=history,
            model=model_override,
            max_tokens=max_tokens,
        )
    except RuntimeError as e:
        logger.error(f"LLM error: {e}")
        return jsonify({
            "error": str(e),
            "hint": "Make sure Ollama is running: `ollama serve` and model is pulled: `ollama pull llama3.2:3b`"
        }), 503

    # Update conversation history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response_text})

    # Keep history bounded
    if len(history) > config.MAX_HISTORY_TURNS * 2:
        history[:] = history[-(config.MAX_HISTORY_TURNS * 2):]

    return jsonify({
        "response": response_text,
        "conversation_id": conv_id,
        "sources": sources,
        "model": model_used,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Health check — returns status of all components."""
    from src.rag.retriever import get_vectorstore
    vs = get_vectorstore()

    ollama_ok = check_ollama_available()
    index_count = vs.count() if vs else 0

    status = {
        "status": "ok",
        "assistant": config.ASSISTANT_NAME,
        "creator": config.CREATOR_NAME,
        "llm_backend": config.LLM_BACKEND,
        "ollama_available": ollama_ok,
        "ollama_model": config.OLLAMA_MODEL,
        "index_documents": index_count,
        "index_ready": index_count > 0,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if not ollama_ok and config.LLM_BACKEND == "ollama":
        status["status"] = "degraded"
        status["warning"] = "Ollama is not running. Start with: `ollama serve`"

    if index_count == 0:
        status["status"] = "degraded"
        status["warning"] = "Vector index is empty. Run: `python scripts/build_index.py`"

    return jsonify(status)


@app.route("/api/info", methods=["GET"])
def info():
    """Returns information about SanchitAI."""
    personal_data_path = config.PERSONAL_DATA_PATH
    identity = {}
    ai_info = {}
    if personal_data_path.exists():
        with open(personal_data_path) as f:
            data = json.load(f)
            identity = data.get("identity", {})
            ai_info = data.get("ai_assistant", {})

    return jsonify({
        "assistant": {
            "name": ai_info.get("name", config.ASSISTANT_NAME),
            "purpose": ai_info.get("purpose", ""),
            "created_by": ai_info.get("created_by", config.CREATOR_NAME),
            "built_with": ai_info.get("built_with", ""),
            "project_type": ai_info.get("project_type", ""),
            "github_repo": ai_info.get("github_repo", ""),
        },
        "creator": {
            "name": identity.get("full_name", config.CREATOR_NAME),
            "website": identity.get("website", ""),
            "github": identity.get("github", ""),
            "linkedin": identity.get("linkedin", ""),
        },
        "api": {
            "chat_endpoint": "POST /api/chat",
            "health_endpoint": "GET /api/health",
            "info_endpoint": "GET /api/info",
        }
    })


@app.route("/api/rebuild-index", methods=["POST"])
@require_api_key
def rebuild_index():
    """
    Rebuild the vector index from scratch.
    Useful after editing personal_data.json or knowledge base files.
    Optionally protected by API key (set API_KEY in .env).
    """
    try:
        from scripts.build_index import build_index
        stats = build_index()
        reload_vectorstore()
        return jsonify({
            "status": "success",
            "message": "Vector index rebuilt successfully.",
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })
    except Exception as e:
        logger.error(f"Index rebuild failed: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Dev server — use Gunicorn/mod_wsgi in production
    port = int(os.getenv("PORT", 5050))
    logger.info(f"Starting SanchitAI dev server on port {port}")
    logger.info(f"LLM Backend: {config.LLM_BACKEND} | Model: {config.OLLAMA_MODEL}")
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "0") == "1")
