"""
RAG Retriever: given a query, retrieves the most relevant context
from the vector store and formats it for the LLM prompt.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.rag.embeddings import embed_query
from src.rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)

_vectorstore: Optional[VectorStore] = None


def get_vectorstore() -> VectorStore:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = VectorStore()
    return _vectorstore


def retrieve(query: str, top_k: int = None) -> List[Dict]:
    """Retrieve top_k most relevant document chunks for a query."""
    top_k = top_k or config.TOP_K_RESULTS
    vs = get_vectorstore()

    if vs.count() == 0:
        logger.warning("Vector store is empty. Run `python scripts/build_index.py` first.")
        return []

    query_embedding = embed_query(query)
    results = vs.query(query_embedding, top_k=top_k)
    logger.debug(f"Retrieved {len(results)} chunks for query: '{query[:60]}'")
    return results


def format_context(retrieved_docs: List[Dict]) -> str:
    """Format retrieved documents into a context block for the LLM."""
    if not retrieved_docs:
        return ""

    context_parts = []
    seen = set()
    i = 1
    for doc in retrieved_docs:
        content = doc["content"].strip()
        if content not in seen:
            seen.add(content)
            source = doc["metadata"].get("source", "unknown")
            section = doc["metadata"].get("section", "")
            label = f"{source} › {section}" if section else source
            context_parts.append(f"[{i}. {label}]\n{content}")
            i += 1

    return "\n\n---\n\n".join(context_parts)


def reload_vectorstore():
    """Force reload the vectorstore (useful after rebuilding index)."""
    global _vectorstore
    _vectorstore = None
    logger.info("Vector store cache cleared — will reload on next query.")
