"""
fastembed-based embedding wrapper.
Uses ONNX runtime — no PyTorch, ~100MB footprint vs ~2.5GB for sentence-transformers.
"""

import logging
from typing import List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)

_model = None


def get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _model = TextEmbedding(model_name=config.EMBEDDING_MODEL)
        logger.info("Embedding model ready.")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]


def embed_query(query: str) -> List[float]:
    model = get_model()
    result = list(model.embed([query]))
    return result[0].tolist()
