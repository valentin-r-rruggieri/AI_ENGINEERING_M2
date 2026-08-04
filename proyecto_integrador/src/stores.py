from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .embeddings import normalize_rows
from .models import RetrievedChunk, StoredChunk


class VectorStoreBackend(Protocol):
    name: str
    def build(self, chunks: list[StoredChunk], vectors: list[list[float]]) -> None: ...
    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]: ...
    def exists(self) -> bool: ...


class FaissVectorStore:
    name = "faiss"

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.index_path = directory / "index.faiss"
        self.records_path = directory / "records.json"
        self._index = None
        self._records: list[StoredChunk] = []

    def exists(self) -> bool:
        return self.index_path.exists() and self.records_path.exists()

    def build(self, chunks: list[StoredChunk], vectors: list[list[float]]) -> None:
        import faiss
        if len(chunks) != len(vectors):
            raise ValueError("Chunks y vectores deben tener la misma cantidad.")
        self.directory.mkdir(parents=True, exist_ok=True)
        matrix = normalize_rows(vectors)
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)
        self._records = chunks
        faiss.write_index(self._index, str(self.index_path))
        self.records_path.write_text(
            json.dumps([item.model_dump() for item in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        import faiss
        if not self.exists():
            raise FileNotFoundError("Índice FAISS inexistente. Ejecutá src.index primero.")
        self._index = faiss.read_index(str(self.index_path))
        self._records = [StoredChunk.model_validate(item) for item in json.loads(
            self.records_path.read_text(encoding="utf-8")
        )]

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        if self._index is None:
            self._load()
        vector = normalize_rows([query_vector])
        scores, positions = self._index.search(vector, top_k)
        return [
            RetrievedChunk(**self._records[position].model_dump(), score=float(score))
            for score, position in zip(scores[0], positions[0]) if position >= 0
        ]


class ChromaVectorStore:
    name = "chroma"

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.collection_name = "peopleflow_faq"

    def _collection(self):
        import chromadb
        client = chromadb.PersistentClient(path=str(self.directory))
        return client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def exists(self) -> bool:
        return self.directory.exists() and any(self.directory.iterdir())

    def build(self, chunks: list[StoredChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunks y vectores deben tener la misma cantidad.")
        self.directory.mkdir(parents=True, exist_ok=True)
        collection = self._collection()
        existing = collection.get()["ids"]
        if existing:
            collection.delete(ids=existing)
        collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=normalize_rows(vectors).tolist(),
            metadatas=[{
                "source": chunk.source,
                "start_word": chunk.start_word,
                "end_word": chunk.end_word,
                **{key: str(value) for key, value in chunk.metadata.items()},
            } for chunk in chunks],
        )

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        if not self.exists():
            raise FileNotFoundError("Índice Chroma inexistente. Ejecutá src.index primero.")
        result = self._collection().query(
            query_embeddings=normalize_rows([query_vector]).tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                content=document,
                source=metadata["source"],
                start_word=int(metadata["start_word"]),
                end_word=int(metadata["end_word"]),
                metadata={key: value for key, value in metadata.items()
                          if key not in {"source", "start_word", "end_word"}},
                score=1 - float(distance),
            )
            for chunk_id, document, metadata, distance in zip(
                result["ids"][0], result["documents"][0],
                result["metadatas"][0], result["distances"][0],
            )
        ]


def create_store(name: str, storage_dir: Path) -> VectorStoreBackend:
    if name == "faiss":
        return FaissVectorStore(storage_dir / "faiss")
    if name == "chroma":
        return ChromaVectorStore(storage_dir / "chroma")
    raise ValueError("backend debe ser 'faiss' o 'chroma'.")

