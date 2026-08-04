from __future__ import annotations

import argparse
from pathlib import Path

from .chunking import split_into_chunks
from .config import Settings
from .embeddings import embed_texts
from .stores import create_store


def load_document(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"No existe el documento: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("El documento fuente está vacío.")
    return text


def build_index(backend: str = "all") -> dict[str, int]:
    settings = Settings.from_env()
    source_path = settings.data_dir / "faq_document.txt"
    chunks = split_into_chunks(
        load_document(source_path), source_path.name,
        settings.chunk_size, settings.chunk_overlap,
    )
    vectors = embed_texts([chunk.content for chunk in chunks], settings)
    if len(chunks) != len(vectors):
        raise AssertionError("La cantidad de chunks y embeddings no coincide.")
    backends = ["chroma", "faiss"] if backend == "all" else [backend]
    for name in backends:
        create_store(name, settings.storage_dir).build(chunks, vectors)
    return {"chunks": len(chunks), "embeddings": len(vectors), "backends": len(backends)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye los índices del FAQ.")
    parser.add_argument("--backend", choices=["all", "chroma", "faiss"], default="all")
    args = parser.parse_args()
    result = build_index(args.backend)
    print(f"Chunks: {result['chunks']} | embeddings: {result['embeddings']} | índices: {result['backends']}")


if __name__ == "__main__":
    main()

