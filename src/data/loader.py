"""
Loads and prepares documents from all knowledge sources:
- data/knowledge_base/*.md  (markdown files)
- data/knowledge_base/*.pdf (PDF files)
- data/personal_data.json   (personal/editable data)
- data/website_pages/publications.json
- data/website_pages/portfolios.json
"""

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import List, Dict, Tuple
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
    """Character-window chunker — used as fallback for oversized sections and PDFs."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                idx = chunk.rfind(sep)
                if idx > chunk_size // 2:
                    chunk = chunk[:idx + len(sep)]
                    break
        chunks.append(chunk.strip())
        # Advance by (chunk - overlap); when chunk ≤ overlap skip the full chunk
        # to avoid getting stuck on separator-free text (e.g. bare URLs)
        advance = len(chunk) - overlap
        start += advance if advance > 0 else len(chunk)
    return [c for c in chunks if len(c) > 30]


def split_markdown_by_headers(text: str) -> List[Tuple[str, str]]:
    """
    Split markdown into (section_title, section_text) pairs on ## headers.
    Sections longer than 1500 chars are further split with chunk_text().
    Returns a flat list of (title, content) tuples ready to become Documents.
    """
    # Find all ## and ### header positions (not #### or deeper)
    header_re = re.compile(r'(?m)^(#{2,3}\s+.+)$')
    positions = [(m.start(), m.group(1).strip()) for m in header_re.finditer(text)]

    chunks: List[Tuple[str, str]] = []

    # Preamble — content before the first ## header
    preamble_end = positions[0][0] if positions else len(text)
    preamble = text[:preamble_end].strip()
    if preamble and len(preamble) > 30:
        h1 = re.match(r'^#\s+(.+)', preamble)
        title = h1.group(1).strip() if h1 else "Overview"
        chunks.extend(_maybe_split(title, preamble))

    for idx, (pos, header_line) in enumerate(positions):
        end = positions[idx + 1][0] if idx + 1 < len(positions) else len(text)
        section_text = text[pos:end].strip()
        section_title = header_line.lstrip('#').strip()

        if not section_text or len(section_text) < 30:
            continue
        chunks.extend(_maybe_split(section_title, section_text))

    # Fallback: no headers found → old chunker
    if not chunks:
        for i, c in enumerate(chunk_text(text)):
            chunks.append(("", c))

    return chunks


def _maybe_split(title: str, text: str) -> List[Tuple[str, str]]:
    """Return one chunk normally; split with chunk_text() if text > 1500 chars."""
    if len(text) <= 1500:
        return [(title, text)]
    return [(title, part) for part in chunk_text(text)]


def load_pdf_files() -> List[Document]:
    docs = []
    kb_dir = config.KNOWLEDGE_BASE_DIR
    if not kb_dir.exists():
        return docs

    for pdf_file in kb_dir.glob("*.pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_file))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
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
        sections = split_markdown_by_headers(text)
        for i, (section_title, section_text) in enumerate(sections):
            docs.append(Document(
                content=section_text,
                metadata={
                    "source": md_file.name,
                    "source_type": "markdown",
                    "section": section_title,
                    "chunk_index": i,
                    "file_path": str(md_file),
                }
            ))
        logger.info(f"Loaded {len(sections)} chunks from {md_file.name}")
    return docs


# ── Website pages: publications & portfolios ──────────────────────────────────

def _strip_html(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text)


def _serialize_publication(entry: dict) -> str:
    parts = [f"Publication: {entry.get('title', '')}"]
    authors = _strip_html(entry.get('authors', ''))
    if authors:
        parts.append(f"Authors: {authors}")
    if entry.get('journal'):
        parts.append(f"Journal: {entry['journal']}")
    if entry.get('year'):
        parts.append(f"Year: {entry['year']}")
    if entry.get('abstract'):
        parts.append(f"Abstract: {entry['abstract']}")
    if entry.get('doi'):
        parts.append(f"DOI/URL: {entry['doi']}")
    return '\n'.join(parts)


def _serialize_portfolio(entry: dict) -> str:
    parts = [f"Project: {entry.get('title', '')}"]
    if entry.get('meta'):
        parts.append(f"Type: {entry['meta']}")
    if entry.get('tagline'):
        parts.append(f"Tagline: {entry['tagline']}")
    if entry.get('impact'):
        parts.append(f"Impact: {entry['impact']}")
    if entry.get('description'):
        parts.append(f"Description: {entry['description']}")
    details = entry.get('details', {})
    if details.get('techStack'):
        parts.append(f"Technologies: {', '.join(details['techStack'])}")
    if details.get('overviewBase'):
        parts.append(f"Overview: {details['overviewBase']}")
    if details.get('challenge'):
        parts.append(f"Challenge: {details['challenge']}")
    if details.get('solution'):
        parts.append(f"Solution: {details['solution']}")
    if details.get('codeLink'):
        parts.append(f"Code repository available on GitHub")
    if details.get('paperLink'):
        parts.append(f"Published in a peer-reviewed journal")
    return '\n'.join(parts)


def load_website_pages() -> List[Document]:
    """Load publications.json and portfolios.json from data/website_pages/."""
    docs = []
    pages_dir = config.WEBSITE_PAGES_DIR
    if not pages_dir.exists():
        logger.warning(f"website_pages directory not found: {pages_dir}")
        return docs

    # ── publications ──────────────────────────────────────────────────────────
    pub_path = pages_dir / "publications.json"
    if pub_path.exists():
        try:
            with open(pub_path, encoding="utf-8") as f:
                publications = json.load(f)
            for i, entry in enumerate(publications):
                text = _serialize_publication(entry)
                if len(text) > 30:
                    docs.append(Document(
                        content=text,
                        metadata={
                            "source": "publications.json",
                            "source_type": "publication",
                            "section": entry.get('title', '')[:80],
                            "chunk_index": i,
                        }
                    ))
            logger.info(f"Loaded {len(publications)} publication chunks from publications.json")
        except Exception as e:
            logger.error(f"Failed to load publications.json: {e}")

    # ── portfolios ────────────────────────────────────────────────────────────
    port_path = pages_dir / "portfolios.json"
    if port_path.exists():
        try:
            with open(port_path, encoding="utf-8") as f:
                portfolios = json.load(f)
            for i, entry in enumerate(portfolios):
                text = _serialize_portfolio(entry)
                if len(text) > 30:
                    # Split large portfolio entries (detailed ones can be ~2k chars)
                    if len(text) > 1500:
                        sub_chunks = chunk_text(text)
                        for j, sub in enumerate(sub_chunks):
                            docs.append(Document(
                                content=sub,
                                metadata={
                                    "source": "portfolios.json",
                                    "source_type": "portfolio",
                                    "section": entry.get('title', '')[:80],
                                    "chunk_index": j,
                                }
                            ))
                    else:
                        docs.append(Document(
                            content=text,
                            metadata={
                                "source": "portfolios.json",
                                "source_type": "portfolio",
                                "section": entry.get('title', '')[:80],
                                "chunk_index": i,
                            }
                        ))
            logger.info(f"Loaded {len(portfolios)} portfolio entries from portfolios.json")
        except Exception as e:
            logger.error(f"Failed to load portfolios.json: {e}")

    return docs


# ── Personal data ─────────────────────────────────────────────────────────────

def _serialize_education(entry: dict) -> str:
    parts = [f"Education: {entry.get('degree', '')}"]
    if entry.get("specialization"):
        parts[0] += f" (specialization: {entry['specialization']})"
    parts.append(f"Institution: {entry.get('institution', '')} — {entry.get('location', '')}")
    parts.append(f"Period: {entry.get('period', '')}")
    if entry.get("gpa"):
        parts.append(f"GPA: {entry['gpa']}")
    if entry.get("achievements"):
        parts.append(f"Achievements: {', '.join(entry['achievements'])}")
    if entry.get("highlights"):
        parts.append(f"Focus areas: {', '.join(entry['highlights'])}")
    return "\n".join(parts)


def _serialize_experience(entry: dict) -> str:
    parts = [f"Experience: {entry.get('title', '')} at {entry.get('organization', '')}"]
    parts.append(f"Period: {entry.get('period', '')} | Type: {entry.get('type', '')}")
    for h in entry.get("highlights", []):
        parts.append(f"- {h}")
    return "\n".join(parts)


def _serialize_project(entry: dict) -> str:
    parts = [f"Project: {entry.get('name', '')}"]
    if entry.get("description"):
        parts.append(entry["description"])
    if entry.get("significance"):
        parts.append(f"Significance: {entry['significance']}")
    if entry.get("impact"):
        parts.append(f"Impact: {entry['impact']}")
    if entry.get("technologies"):
        parts.append(f"Technologies: {', '.join(entry['technologies'])}")
    if entry.get("github"):
        parts.append(f"GitHub: {entry['github']}")
    return "\n".join(parts)


def flatten_json_to_text(data, prefix="") -> List[str]:
    """Recursively convert JSON to readable text passages."""
    passages = []

    if isinstance(data, dict):
        if data.get("_instructions"):
            return passages

        if "question" in data and "answer" in data:
            passages.append(f"Q: {data['question']}\nA: {data['answer']}")
            return passages

        if "title" in data and "story" in data:
            passages.append(f"{data['title']}: {data['story']}")
            return passages

        if "degree" in data and "institution" in data:
            passages.append(_serialize_education(data))
            return passages

        if "title" in data and "organization" in data:
            passages.append(_serialize_experience(data))
            return passages

        if "name" in data and "description" in data and "technologies" in data:
            passages.append(_serialize_project(data))
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

    # Computed age
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
        except Exception as e:
            logger.warning(f"Could not compute age from date_of_birth '{dob_str}': {e}")

    # FAQ entries
    faqs = data.get("frequently_asked_questions", [])
    for faq in faqs:
        if isinstance(faq, dict) and faq.get("question") and not faq["question"].startswith("ADD_"):
            docs.append(Document(
                content=f"Q: {faq['question']}\nA: {faq['answer']}",
                metadata={"source": "personal_data.json", "source_type": "faq", "chunk_index": 0}
            ))

    # AI assistant identity
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

    # All other sections
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
    all_docs.extend(load_website_pages())
    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs
