from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from build_index import load_documents, split_documents
from config import Settings


def test_corpus_generates_traceable_chunks() -> None:
    settings = Settings(
        embedding_model="text-embedding-3-small",
        chat_model="gpt-4o-mini",
        chunk_size=300,
        chunk_overlap=50,
        top_k=4,
        collection_name="test_collection",
    )
    chunks = split_documents(load_documents(), settings)

    assert len(chunks) >= 20
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(chunk.metadata["source"] == "faq_document.txt" for chunk in chunks)
