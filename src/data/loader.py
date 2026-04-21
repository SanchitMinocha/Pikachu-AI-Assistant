"""
Loads and prepares documents from all knowledge sources:
- data/knowledge_base/*.md  (markdown files)
- data/personal_data.json   (personal/editable data)
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)


class Document:
    def __init__(self, content: str, metadata: Dict):
        self.content = content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(source={self.metadata.get('source')}, len={len(self.content)})"


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # prefer splitting at paragraph or sentence boundary
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                idx = chunk.rfind(sep)
                if idx > chunk_size // 2:
                    chunk = chunk[:idx + len(sep)]
                    break
        chunks.append(chunk.strip())
        start += max(1, len(chunk) - overlap)
    return [c for c in chunks if len(c) > 30]


def load_pdf_files() -> List[Document]:
    docs = []
    kb_dir = config.KNOWLEDGE_BASE_DIR
    if not kb_dir.exists():
        return docs

    for pdf_file in kb_dir.glob("*.pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_file))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            text = text.strip()
            if not text:
                logger.warning(f"No text extracted from {pdf_file.name}")
                continue
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                docs.append(Document(
                    content=chunk,
                    metadata={
                        "source": pdf_file.name,
                        "source_type": "pdf",
                        "chunk_index": i,
                        "file_path": str(pdf_file),
                    }
                ))
            logger.info(f"Loaded {len(chunks)} chunks from {pdf_file.name}")
        except ImportError:
            logger.error("pypdf not installed — run: pip install pypdf>=4.0.0")
        except Exception as e:
            logger.error(f"Failed to load {pdf_file.name}: {e}")
    return docs


def load_markdown_files() -> List[Document]:
    docs = []
    kb_dir = config.KNOWLEDGE_BASE_DIR
    if not kb_dir.exists():
        logger.warning(f"Knowledge base directory not found: {kb_dir}")
        return docs

    for md_file in kb_dir.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                content=chunk,
                metadata={
                    "source": md_file.name,
                    "source_type": "markdown",
                    "chunk_index": i,
                    "file_path": str(md_file),
                }
            ))
        logger.info(f"Loaded {len(chunks)} chunks from {md_file.name}")
    return docs


def flatten_json_to_text(data, prefix="") -> List[str]:
    """Recursively convert JSON to readable text passages."""
    passages = []

    if isinstance(data, dict):
        if data.get("_instructions"):
            return passages  # skip meta key

        # Handle FAQ entries
        if "question" in data and "answer" in data:
            passages.append(f"Q: {data['question']}\nA: {data['answer']}")
            return passages

        # Handle story entries
        if "title" in data and "story" in data:
            passages.append(f"{data['title']}: {data['story']}")
            return passages

        for k, v in data.items():
            if k.startswith("_"):
                continue
            new_prefix = f"{prefix} > {k}".strip(" >")
            passages.extend(flatten_json_to_text(v, new_prefix))

    elif isinstance(data, list):
        for item in data:
            passages.extend(flatten_json_to_text(item, prefix))

    elif isinstance(data, str) and data and not data.startswith("ADD_"):
        if prefix:
            passages.append(f"{prefix}: {data}")
        else:
            passages.append(data)

    elif isinstance(data, (int, float)) and data is not None:
        if prefix:
            passages.append(f"{prefix}: {data}")

    return passages


def load_personal_data() -> List[Document]:
    docs = []
    path = config.PERSONAL_DATA_PATH
    if not path.exists():
        logger.warning(f"personal_data.json not found: {path}")
        return docs

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Compute and inject age from date_of_birth so the model doesn't have to do math
    identity = data.get("identity", {})
    dob_str = identity.get("date_of_birth", "")
    if dob_str and not dob_str.startswith("ADD_"):
        try:
            from datetime import datetime
            dob = None
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y", "%d %B %Y"):
                try:
                    dob = datetime.strptime(dob_str, fmt).date()
                    break
                except ValueError:
                    continue
            if dob is None:
                raise ValueError(f"Unrecognised date format: {dob_str}")
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            age_fact = (
                f"Sanchit Minocha's date of birth is {dob_str}. "
                f"He is currently {age} years old (as of {today.strftime('%B %Y')})."
            )
            docs.append(Document(
                content=age_fact,
                metadata={"source": "personal_data.json", "source_type": "computed_age", "chunk_index": 0}
            ))
            logger.info(f"Computed age: {age} from DOB: {dob_str}")
        except Exception as e:
            logger.warning(f"Could not compute age from date_of_birth '{dob_str}': {e}")

    # Load FAQ entries as individual documents (highest priority)
    faqs = data.get("frequently_asked_questions", [])
    for faq in faqs:
        if isinstance(faq, dict) and faq.get("question") and not faq["question"].startswith("ADD_"):
            docs.append(Document(
                content=f"Q: {faq['question']}\nA: {faq['answer']}",
                metadata={"source": "personal_data.json", "source_type": "faq", "chunk_index": 0}
            ))

    # Load AI assistant identity
    ai_info = data.get("ai_assistant", {})
    if ai_info:
        text = (
            f"About {ai_info.get('name', 'Pikachu - Sanchit\'s AI Assistant')}: {ai_info.get('purpose', '')}\n"
            f"Built by: {ai_info.get('created_by', '')}\n"
            f"Built with: {ai_info.get('built_with', '')}\n"
            f"Project type: {ai_info.get('project_type', '')}"
        )
        docs.append(Document(
            content=text,
            metadata={"source": "personal_data.json", "source_type": "identity", "chunk_index": 0}
        ))

    # Convert all sections to text passages
    for section_key, section_data in data.items():
        if section_key in ("_instructions", "frequently_asked_questions", "ai_assistant"):
            continue
        passages = flatten_json_to_text(section_data, prefix=section_key.replace("_", " ").title())
        for i, passage in enumerate(passages):
            if len(passage) > 20:
                docs.append(Document(
                    content=passage,
                    metadata={
                        "source": "personal_data.json",
                        "source_type": "personal_data",
                        "section": section_key,
                        "chunk_index": i,
                    }
                ))

    logger.info(f"Loaded {len(docs)} documents from personal_data.json")
    return docs


def load_all_documents() -> List[Document]:
    """Load all documents from all sources."""
    all_docs = []
    all_docs.extend(load_markdown_files())
    all_docs.extend(load_pdf_files())
    all_docs.extend(load_personal_data())
    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs
