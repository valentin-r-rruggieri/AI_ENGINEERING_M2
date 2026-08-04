from __future__ import annotations

import re

from .models import StoredChunk


def clean_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        raise ValueError("El documento está vacío después de normalizarlo.")
    return cleaned


def split_into_chunks(text: str, source: str, chunk_size: int, overlap: int) -> list[StoredChunk]:
    if not 50 <= chunk_size <= 500:
        raise ValueError("chunk_size debe estar entre 50 y 500 palabras.")
    if not 0 < overlap < chunk_size:
        raise ValueError("chunk_overlap debe ser positivo y menor a chunk_size.")
    words = clean_text(text).split()
    chunks: list[StoredChunk] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        content = " ".join(words[start:end])
        chunks.append(StoredChunk(
            chunk_id=f"faq-{len(chunks) + 1:03d}",
            content=content,
            source=source,
            start_word=start,
            end_word=end,
            metadata={"word_count": end - start, "version": "1"},
        ))
        if end == len(words):
            break
        start = end - overlap
    if len(chunks) < 20:
        raise ValueError(f"Se requieren al menos 20 chunks; se generaron {len(chunks)}.")
    return chunks

