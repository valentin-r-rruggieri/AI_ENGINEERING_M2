"""Configuración compartida del proyecto."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "faq_document.txt"
CHROMA_DIR = BASE_DIR / "storage" / "chroma"


@dataclass(frozen=True)
class Settings:
    """Parámetros necesarios para indexar y consultar."""

    embedding_model: str
    chat_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    collection_name: str


def load_settings() -> Settings:
    """Carga el archivo .env y devuelve la configuración validada."""
    load_dotenv(BASE_DIR / ".env")

    chunk_size = int(os.getenv("CHUNK_SIZE", "300"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k = int(os.getenv("TOP_K", "4"))

    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE debe ser mayor que cero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP debe ser mayor o igual a cero y menor que CHUNK_SIZE.")
    if not 2 <= top_k <= 5:
        raise ValueError("TOP_K debe estar entre 2 y 5.")

    return Settings(
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        collection_name=os.getenv("CHROMA_COLLECTION", "peopleflow_faq"),
    )


def require_openai_api_key() -> None:
    """Falla temprano si falta la credencial necesaria para los modelos OpenAI."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "Falta OPENAI_API_KEY. Copiá .env.example a .env y configurá tu clave."
        )
