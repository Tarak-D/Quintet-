"""
utils/rag.py

Grounds tutor explanations and generated questions in a curated knowledge
base (proposal 3.1 / 6). Uses Chroma with a local embedding function so the
hackathon MVP doesn't need an external embeddings API.

Layout expected under /knowledge_base:
    knowledge_base/
        fractions/
            common_denominators.md
            adding_fractions.md
        ...
Each subfolder name = a topic id used elsewhere in the system.
"""

import os
import glob
import logging
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger("eduleap.rag")

KB_DIR = os.getenv("EDULEAP_KB_DIR", "./knowledge_base")
CHROMA_DIR = os.getenv("EDULEAP_CHROMA_DIR", "./.chroma")

_embed_fn = embedding_functions.DefaultEmbeddingFunction()  # local, no API key needed
_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(name="eduleap_kb", embedding_function=_embed_fn)


def index_knowledge_base(kb_dir: str = KB_DIR) -> int:
    """
    (Re)builds the vector store from the markdown/text files in knowledge_base/.
    Run once at startup or whenever content changes. Returns number of chunks indexed.
    """
    files = glob.glob(os.path.join(kb_dir, "**", "*.md"), recursive=True) + \
        glob.glob(os.path.join(kb_dir, "**", "*.txt"), recursive=True)

    if not files:
        logger.warning("No knowledge base files found under %s", kb_dir)
        return 0

    ids, docs, metadatas = [], [], []
    for path in files:
        topic = os.path.relpath(path, kb_dir).split(os.sep)[0]
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # naive chunking: split on blank lines, merge small chunks
        chunks, buf = [], ""
        for para in content.split("\n\n"):
            buf += para + "\n\n"
            if len(buf) > 600:
                chunks.append(buf.strip())
                buf = ""
        if buf.strip():
            chunks.append(buf.strip())

        for i, chunk in enumerate(chunks):
            ids.append(f"{path}::{i}")
            docs.append(chunk)
            metadatas.append({"topic": topic, "source": path})

    if ids:
        _collection.upsert(ids=ids, documents=docs, metadatas=metadatas)

    logger.info("Indexed %d chunks from %d files", len(ids), len(files))
    return len(ids)


def retrieve_context(query: str, topic: Optional[str] = None, k: int = 4) -> str:
    """
    Returns concatenated top-k passages relevant to `query`, optionally
    filtered to a topic, formatted for direct insertion into an LLM prompt.
    """
    where = {"topic": topic} if topic else None
    try:
        results = _collection.query(query_texts=[query], n_results=k, where=where)
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG query failed (%s); returning empty context", exc)
        return ""

    docs = results.get("documents", [[]])[0]
    if not docs:
        return ""

    return "\n\n---\n\n".join(docs)
