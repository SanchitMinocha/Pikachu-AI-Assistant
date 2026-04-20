"""
ChromaDB-based persistent vector store for SanchitAI.
Handles document insertion, persistence, and similarity search.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not installed. Run: pip install chromadb")


COLLECTION_NAME = "sanchitai_knowledge"


class VectorStore:
    def __init__(self):
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is required. Install with: pip install chromadb")

        config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(config.VECTORSTORE_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            # Collection was deleted/rebuilt while this process was running — reset cache
            self._collection = None
            return self.collection.count()

    def clear(self):
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self._collection = None
        except Exception:
            pass

    def add_documents(self, documents, embeddings: List[List[float]]):
        """Add documents with pre-computed embeddings."""
        ids = [f"doc_{i}" for i in range(len(documents))]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # ChromaDB has a batch size limit — insert in chunks
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                documents=contents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )
        logger.info(f"Added {len(ids)} documents to vector store")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = None,
    ) -> List[Dict]:
        """Return top_k most similar documents."""
        top_k = top_k or config.TOP_K_RESULTS
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # Stale collection reference — reset and retry once
            self._collection = None
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.count()),
                include=["documents", "metadatas", "distances"],
            )
        docs = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1 - dist   # cosine distance → similarity
            if similarity >= config.SIMILARITY_THRESHOLD:
                docs.append({
                    "content": doc,
                    "metadata": meta,
                    "similarity": round(similarity, 4),
                })
        return docs
