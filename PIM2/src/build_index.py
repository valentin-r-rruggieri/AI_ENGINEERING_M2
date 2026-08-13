"""Construye el índice vectorial persistente de PeopleFlow."""
from __future__ import annotations

import sys

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_DIR, DATA_PATH, Settings, load_settings, require_openai_api_key
from utils import clean_text, ensure_file_exists


def load_documents():
    """Carga el FAQ y registra la fuente en su metadata."""
    ensure_file_exists(DATA_PATH)
    documents = TextLoader(str(DATA_PATH), encoding="utf-8").load()
    for document in documents:
        document.page_content = clean_text(document.page_content)
        document.metadata["source"] = DATA_PATH.name
    return documents


def split_documents(documents, settings: Settings):
    """Fragmenta el corpus y agrega identificadores trazables a cada chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    if len(chunks) < 20:
        raise ValueError(
            f"Se generaron {len(chunks)} chunks; el proyecto requiere al menos 20."
        )
    for number, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = number
    return chunks


def build_vector_store(chunks, settings: Settings) -> Chroma:
    """Recrea la colección local y almacena un embedding por cada chunk."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    vector_store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    vector_store.reset_collection()
    vector_store.add_documents(chunks)
    stored_count = vector_store._collection.count()
    if stored_count != len(chunks):
        raise RuntimeError(
            f"El índice guardó {stored_count} vectores para {len(chunks)} chunks."
        )
    return vector_store


def main() -> None:
    """Ejecuta el pipeline de indexación desde la consola."""
    try:
        settings = load_settings()
        require_openai_api_key()
        documents = load_documents()
        chunks = split_documents(documents, settings)
        build_vector_store(chunks, settings)
    except (EnvironmentError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error de indexación: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("Documento cargado correctamente.")
    print(f"Chunks generados: {len(chunks)}")
    print(f"Embeddings generados: {len(chunks)}")
    print(f"Índice guardado correctamente en: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
