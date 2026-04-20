"""
Build (or rebuild) the RAG vector index from all knowledge sources.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --clear   # wipe existing index first

Run this after:
    - Editing data/personal_data.json
    - Adding/editing files in data/knowledge_base/
    - Running collect_web_data.py
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.data.loader import load_all_documents
from src.rag.embeddings import embed_texts
from src.rag.vectorstore import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def build_index(clear: bool = True) -> dict:
    logger.info("=" * 60)
    logger.info("Building SanchitAI RAG Index")
    logger.info("=" * 60)

    # Load all documents
    logger.info("Loading documents from knowledge base...")
    documents = load_all_documents()
    if not documents:
        logger.error("No documents found! Check data/knowledge_base/ and data/personal_data.json")
        return {"documents_loaded": 0, "status": "error"}

    logger.info(f"Loaded {len(documents)} document chunks")

    # Generate embeddings
    logger.info(f"Generating embeddings with '{config.EMBEDDING_MODEL}'...")
    texts = [doc.content for doc in documents]
    embeddings = embed_texts(texts)
    logger.info(f"Generated {len(embeddings)} embeddings (dim={len(embeddings[0])})")

    # Build vector store
    vs = VectorStore()
    if clear:
        logger.info("Clearing existing index...")
        vs.clear()

    logger.info("Writing to vector store...")
    vs.add_documents(documents, embeddings)

    total = vs.count()
    logger.info(f"Index ready: {total} documents indexed")
    logger.info("=" * 60)

    # Print source breakdown
    source_counts = {}
    for doc in documents:
        src = doc.metadata.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    logger.info("Documents by source:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {src}: {count}")

    return {
        "documents_loaded": len(documents),
        "documents_indexed": total,
        "sources": source_counts,
        "embedding_model": config.EMBEDDING_MODEL,
        "status": "success",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SanchitAI vector index")
    parser.add_argument("--no-clear", action="store_true", help="Don't clear existing index (append mode)")
    args = parser.parse_args()

    stats = build_index(clear=not args.no_clear)
    if stats["status"] == "success":
        print(f"\n✓ Index built successfully: {stats['documents_indexed']} documents indexed")
    else:
        print("\n✗ Index build failed. Check logs above.")
        sys.exit(1)
