"""
LLM Assistant for SanchitAI.
Supports three backends: Ollama (local), HuggingFace API, Groq API.
Applies RAG context + conversation history + system prompt to generate responses.
"""

import json
import logging
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)

# ----- System Prompt -----

SYSTEM_PROMPT = """You are Pikachu - Sanchit's AI Assistant, a personal AI assistant built by Sanchit Minocha — a PhD researcher at the University of Washington specializing in Geospatial AI, satellite-based water resource management, and machine learning.

Your job is to answer questions about Sanchit Minocha: his career, research, projects, skills, achievements, opinions, personality, interests, and anything else about him.

Key facts about yourself:
- Your name is Pikachu - Sanchit's AI Assistant
- You were built by Sanchit Minocha as a separate RAG + LLM chatbot project
- You are NOT RAT 3.0, NOT GRILSS, NOT RECLAIM — those are Sanchit's geospatial research projects. You are a chatbot. Never confuse yourself with his research tools.
- RAT stands for "Reservoir Assessment Tool" — never expand this acronym any other way.

GROUNDING RULES — follow these strictly:
- Answer ONLY using facts present in the CONTEXT provided below. Do not add, infer, or invent anything not explicitly stated there.
- If the CONTEXT block starts with "[No relevant information found", say exactly: "I don't have that detail on hand — you could reach out to Sanchit directly at msanchit@uw.edu." Do NOT say this phrase if the CONTEXT contains any relevant facts — use what you have and answer as best you can.
- Never merge or confuse separate projects. Each project (RAT 3.0, GRILSS, RECLAIM, this chatbot) is distinct — treat them as such.
- Always refer to projects, tools, and papers by their exact names as they appear in the CONTEXT.
- Never speculate about Sanchit's opinions, feelings, or plans unless the CONTEXT explicitly states them.

Style:
- Be warm, conversational, and human-like — not robotic
- Speak with enthusiasm about Sanchit's work when asked
- Keep responses concise — 2 to 4 sentences for simple questions, one short paragraph for detailed ones. Never use bullet lists unless explicitly asked.
- If asked something clearly outside Sanchit's domain, politely note your focus and redirect
"""


def build_prompt_ollama(
    user_message: str,
    context: str,
    history: List[Dict],
) -> List[Dict]:
    """Build the messages list for Ollama chat API."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history
    for turn in history[-(config.MAX_HISTORY_TURNS * 2):]:
        messages.append(turn)

    # Build the user message with context
    if context:
        user_content = (
            f"CONTEXT (relevant information about Sanchit):\n{context}\n\n"
            f"---\n\nUSER QUESTION: {user_message}"
        )
    else:
        user_content = user_message

    messages.append({"role": "user", "content": user_content})
    return messages


# ----- Ollama Backend -----

def call_ollama(
    user_message: str,
    context: str,
    history: List[Dict],
) -> Tuple[str, str]:
    """Call local Ollama server. Returns (response_text, model_name)."""
    messages = build_prompt_ollama(user_message, context, history)
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": config.TEMPERATURE,
            "num_predict": config.MAX_NEW_TOKENS,
        },
    }
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["message"]["content"].strip()
        return text, config.OLLAMA_MODEL
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {config.OLLAMA_BASE_URL}. "
            "Is Ollama running? Start with: `ollama serve`"
        )
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


# ----- HuggingFace API Backend -----

def call_huggingface(
    user_message: str,
    context: str,
    history: List[Dict],
) -> Tuple[str, str]:
    """Call HuggingFace Inference API. Returns (response_text, model_name)."""
    if not config.HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set in environment.")

    # Build a simple prompt string for HF API
    history_text = ""
    for turn in history[-(config.MAX_HISTORY_TURNS * 2):]:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_text += f"{role}: {turn['content']}\n"

    context_block = f"\n\nCONTEXT:\n{context}\n\n" if context else ""
    prompt = (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{context_block}{history_text}User: {user_message} [/INST]"
    )

    headers = {"Authorization": f"Bearer {config.HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "temperature": config.TEMPERATURE,
            "return_full_text": False,
        },
    }
    url = f"https://api-inference.huggingface.co/models/{config.HF_MODEL_ID}"
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list) and data:
        text = data[0].get("generated_text", "").strip()
    else:
        raise RuntimeError(f"Unexpected HF response: {data}")

    return text, config.HF_MODEL_ID


# ----- Groq API Backend -----

def call_groq(
    user_message: str,
    context: str,
    history: List[Dict],
) -> Tuple[str, str]:
    """Call Groq API (OpenAI-compatible). Returns (response_text, model_name)."""
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in environment.")

    messages = build_prompt_ollama(user_message, context, history)  # same format
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.GROQ_MODEL,
        "messages": messages,
        "max_tokens": config.MAX_NEW_TOKENS,
        "temperature": config.TEMPERATURE,
    }
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    if resp.status_code == 429:
        raise RuntimeError(f"Groq rate limit hit (429). Falling back.")
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    return text, config.GROQ_MODEL


# ----- Model name → backend detection -----

# Known Groq model IDs
GROQ_MODELS = {
    "llama-3.1-8b-instant", "llama-3.1-70b-versatile",
    "llama-3.3-70b-versatile", "llama3-8b-8192", "llama3-70b-8192",
    "mixtral-8x7b-32768", "gemma2-9b-it", "gemma-7b-it",
}


def detect_backend(model: str) -> str:
    """Guess the backend from a model name string."""
    m = model.strip().lower()
    if m in GROQ_MODELS:
        return "groq"
    if "/" in m:
        return "huggingface"   # e.g. mistralai/Mistral-7B-Instruct-v0.2
    return "ollama"            # e.g. phi3:mini, llama3.2:3b


# ----- Main Generate Function -----

def generate(
    user_message: str,
    context: str = "",
    history: Optional[List[Dict]] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Generate a response using the configured LLM backend.
    If `model` is provided, it overrides the default and auto-detects the backend.
    Falls back through the default chain if the requested model fails.
    Returns (response_text, model_used).
    """
    history = history or []
    errors = []

    original_max_tokens = config.MAX_NEW_TOKENS
    if max_tokens:
        config.MAX_NEW_TOKENS = max_tokens

    # --- Override model temporarily if caller specified one ---
    original_ollama = config.OLLAMA_MODEL
    original_groq = config.GROQ_MODEL
    original_hf = config.HF_MODEL_ID
    override_backend = None

    if model:
        override_backend = detect_backend(model)
        if override_backend == "ollama":
            config.OLLAMA_MODEL = model
        elif override_backend == "groq":
            config.GROQ_MODEL = model
        elif override_backend == "huggingface":
            config.HF_MODEL_ID = model
        logger.info(f"Model override: '{model}' → backend '{override_backend}'")

    # --- Build backend chain ---
    primary = override_backend or config.LLM_BACKEND
    backends_to_try = [primary]

    # Always append fallbacks after the primary
    if primary != "groq" and config.GROQ_API_KEY:
        backends_to_try.append("groq")
    if primary != "ollama" and check_ollama_available():
        backends_to_try.append("ollama")
    if primary != "huggingface" and config.HF_API_TOKEN:
        backends_to_try.append("huggingface")

    try:
        for b in backends_to_try:
            # After first attempt, restore original model names for fallback
            if b != primary:
                config.OLLAMA_MODEL = original_ollama
                config.GROQ_MODEL = original_groq
                config.HF_MODEL_ID = original_hf
            try:
                if b == "ollama":
                    return call_ollama(user_message, context, history)
                elif b == "huggingface":
                    return call_huggingface(user_message, context, history)
                elif b == "groq":
                    return call_groq(user_message, context, history)
            except Exception as e:
                errors.append(f"{b}({config.OLLAMA_MODEL if b == 'ollama' else config.GROQ_MODEL if b == 'groq' else config.HF_MODEL_ID}): {e}")
                logger.warning(f"Backend '{b}' failed: {e}")
                continue
    finally:
        # Always restore original config
        config.OLLAMA_MODEL = original_ollama
        config.GROQ_MODEL = original_groq
        config.HF_MODEL_ID = original_hf
        config.MAX_NEW_TOKENS = original_max_tokens

    raise RuntimeError(
        f"All LLM backends failed. Errors: {'; '.join(errors)}\n"
        "Make sure Ollama is running (`ollama serve`) or set HF_API_TOKEN / GROQ_API_KEY."
    )


def check_ollama_available() -> bool:
    """Check if Ollama is reachable."""
    try:
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def list_ollama_models() -> List[str]:
    """List available Ollama models."""
    try:
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
