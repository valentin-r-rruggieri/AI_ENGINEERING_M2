from __future__ import annotations

import numpy as np
from openai import OpenAI

from .config import Settings, api_key


def embed_texts(texts: list[str], settings: Settings) -> list[list[float]]:
    if not texts:
        return []
    client = OpenAI(api_key=api_key())
    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    vectors = [item.embedding for item in response.data]
    if len(vectors) != len(texts):
        raise RuntimeError("La API no devolvió un embedding por cada texto.")
    return vectors


def normalize_rows(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("Los embeddings deben ser una matriz no vacía.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Se recibió un embedding nulo.")
    return matrix / norms

