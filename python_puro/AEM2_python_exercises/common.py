"""Utilidades compartidas para los ejercicios de AEM2."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from dotenv import load_dotenv


@dataclass
class Chunk:
    chunk_id: str
    content: str
    source: str
    start_word: int
    end_word: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_environment() -> None:
    load_dotenv(Path(__file__).with_name(".env"))


def setting(name: str, default: str) -> str:
    load_environment()
    return os.getenv(name, default)


def require_openai_key() -> str:
    load_environment()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError(
            "Falta OPENAI_API_KEY. Copiá .env.example a .env y completá la clave."
        )
    return key


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    a, b = np.asarray(list(left), dtype=float), np.asarray(list(right), dtype=float)
    if a.shape != b.shape or not len(a):
        raise ValueError("Los vectores deben tener la misma dimensión y no estar vacíos.")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        raise ValueError("No se puede calcular coseno con un vector nulo.")
    return float(np.dot(a, b) / denominator)


def normalize(vector: Iterable[float]) -> np.ndarray:
    value = np.asarray(list(vector), dtype=np.float32)
    norm = np.linalg.norm(value)
    if norm == 0:
        raise ValueError("No se puede normalizar un vector nulo.")
    return value / norm


def split_words(text: str, chunk_size: int = 120, overlap: int = 24, source: str = "local") -> list[Chunk]:
    if not text or not text.strip():
        raise ValueError("El texto no puede estar vacío.")
    if not 1 <= overlap < chunk_size:
        raise ValueError("overlap debe ser positivo y menor que chunk_size.")
    words = re.findall(r"\S+", text)
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        content = " ".join(words[start:end])
        chunks.append(Chunk(
            chunk_id=f"{source}-{len(chunks) + 1:03d}",
            content=content,
            source=source,
            start_word=start,
            end_word=end,
            metadata={"chunk_size_words": end - start},
        ))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def deterministic_embedding(text: str, dimensions: int = 32) -> list[float]:
    """Embedding local determinista para tests; no reemplaza un modelo semántico."""
    tokens = re.findall(r"[a-záéíóúñ0-9]+", text.lower())
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        vector[int.from_bytes(digest[:2], "big") % dimensions] += 1
    if not np.any(vector):
        vector[0] = 1
    return normalize(vector).tolist()


def embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    if not texts:
        return []
    client = OpenAI(api_key=require_openai_key())
    response = client.embeddings.create(
        model=setting("AEM2_EMBEDDING_MODEL", "text-embedding-3-small"),
        input=texts,
    )
    return [item.embedding for item in response.data]


def top_k(query: Iterable[float], chunks: list[Chunk], vectors: list[list[float]], k: int = 3) -> list[dict[str, Any]]:
    if len(chunks) != len(vectors):
        raise ValueError("Cada chunk debe tener exactamente un embedding.")
    if not 1 <= k <= len(chunks):
        raise ValueError("k debe estar entre 1 y la cantidad de chunks.")
    scored = [
        {"chunk": chunk.to_dict(), "score": cosine_similarity(query, vector)}
        for chunk, vector in zip(chunks, vectors)
    ]
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:k]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def timed(callable_: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = callable_(*args, **kwargs)
    return result, (time.perf_counter() - started) * 1_000

