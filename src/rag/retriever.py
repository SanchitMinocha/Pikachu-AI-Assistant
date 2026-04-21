"""
RAG Retriever: given a query, retrieves the most relevant context
from the vector store and formats it for the LLM prompt.
"""

import logging
import re
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


def build_retrieval_query(message: str, history: List[Dict] = None) -> str:
    """
    Produce an embedding-optimised retrieval query from a raw user message.

    The user message often contains the person's name ("Is Sanchit Minocha...?")
    and question scaffolding ("Does he have...?") which pull the embedding toward
    generic identity chunks rather than topically relevant content. Stripping them
    forces the embedding to focus on the actual subject being asked about.

    Recent conversation turns are prepended for follow-up question resolution.
    """
    q = message.strip()

    # Remove the person's name variants
    q = re.sub(r'\bsanchit\s+minocha\b', '', q, flags=re.IGNORECASE)
    q = re.sub(r'\bsanchit\b', '', q, flags=re.IGNORECASE)

    # Remove question intro phrases (order matters — longer phrases first)
    q = re.sub(
        r'^(tell me (more about|about)|can you (tell me about|describe|explain|elaborate on)|'
        r'what (is|are|does|did|was|were|has|have)|'
        r'who (is|are|was|were)|how (does|did|is|are|was|were)|'
        r'where (does|did|is|are)|'
        r'does he (have|has|work|do)|is there|are there|'
        r'is|are|do|did|does|has|have|can|'
        r'(please )?(describe|explain|summarize|list))\b\s*',
        '',
        q,
        flags=re.IGNORECASE,
    )

    # Strip leading noise determiners / pronouns
    q = re.sub(r'^(his|he|her|she|their|any|the|a|an)\s+', '', q, flags=re.IGNORECASE)

    q = re.sub(r'\s+', ' ', q).strip(' ?,.')

    # Fall back to full original message if we stripped too much
    retrieval_q = q if len(q) >= 4 else message.strip()

    # Only prepend history when the query is a genuine follow-up:
    #   (a) contains a dangling pronoun with no self-contained topic, or
    #   (b) is an affirmative / continuation phrase like "yes", "tell me more"
    # Self-contained queries ("most impactful work?", "RAT 3.0") must stay clean.
    _dangling = re.compile(
        r'\b(it|that|those|they|them|this|these|the same|the project|the paper|the tool)\b',
        re.IGNORECASE,
    )
    _continuation = re.compile(
        r'^(yes|yeah|yep|yup|sure|ok|okay|definitely|absolutely|'
        r'great|cool|interesting|nice|wow|'
        r'tell me more|go on|more details|elaborate|continue|'
        r'what else|anything else|something more|more|'
        r'and(\s+what)?|really|sounds good)[\s!?.]*$',
        re.IGNORECASE,
    )
    is_followup = _dangling.search(retrieval_q) or _continuation.match(retrieval_q.strip())
    if history and is_followup:
        # Include the last user question for topic context, plus the last assistant
        # response (truncated) so the embedding moves toward content NOT yet covered.
        last_user = next(
            (t["content"][:80] for t in reversed(history) if t.get("role") == "user"), ""
        )
        last_assistant = next(
            (t["content"][:120] for t in reversed(history) if t.get("role") == "assistant"), ""
        )
        context_prefix = " ".join(filter(None, [last_user, last_assistant]))
        if context_prefix:
            retrieval_q = f"{context_prefix} {retrieval_q}"

    logger.debug(f"Retrieval query: {repr(retrieval_q[:120])} (from: {repr(message[:80])})")
    return retrieval_q


def retrieve(query: str, top_k: int = None) -> List[Dict]:
    """Retrieve top_k most relevant document chunks for a query."""
    top_k = top_k or config.TOP_K_RESULTS
    vs = get_vectorstore()

    if vs.count() == 0:
        logger.warning("Vector store is empty. Run `python scripts/build_index.py` first.")
        return []

    query_embedding = embed_query(query)
    results = vs.query(query_embedding, top_k=top_k)

    # Always log retrieved chunks so retrieval failures are visible in logs
    if results:
        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:80]}'")
        for i, doc in enumerate(results):
            source = doc["metadata"].get("source", "unknown")
            section = doc["metadata"].get("section", "")
            label = f"{source} > {section}" if section else source
            preview = doc['content'][:120].replace('\n', ' ').encode('ascii', 'replace').decode('ascii')
            logger.info(
                f"  [{i+1}] score={doc['similarity']:.4f} | {label} | {preview!r}"
            )
    else:
        logger.info(
            f"No chunks passed similarity threshold ({config.SIMILARITY_THRESHOLD}) "
            f"for query: '{query[:80]}'"
        )

    return results


def format_context(retrieved_docs: List[Dict]) -> str:
    """
    Format retrieved documents into a context block for the LLM.
    Returns an explicit 'no context' marker when nothing was retrieved so the
    LLM is told directly to say it doesn't have the information rather than
    falling back to pretrained knowledge.
    """
    if not retrieved_docs:
        return "[No relevant information found in the knowledge base for this query.]"

    context_parts = []
    seen = set()
    i = 1
    for doc in retrieved_docs:
        content = doc["content"].strip()
        if content not in seen:
            seen.add(content)
            source = doc["metadata"].get("source", "unknown")
            section = doc["metadata"].get("section", "")
            score = doc.get("similarity", 0)
            label = f"{source} > {section}" if section else source
            context_parts.append(f"[{i}. {label} (score: {score:.2f})]\n{content}")
            i += 1

    return "\n\n---\n\n".join(context_parts)


def reload_vectorstore():
    """Force reload the vectorstore (useful after rebuilding index)."""
    global _vectorstore
    _vectorstore = None
    logger.info("Vector store cache cleared - will reload on next query.")
