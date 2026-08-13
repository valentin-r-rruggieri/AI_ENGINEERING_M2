"""Funciones puras de validación y construcción de respuestas."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


EXPECTED_RESPONSE_KEYS = {"user_question", "system_answer", "chunks_related"}


def clean_text(text: str) -> str:
    """Elimina caracteres nulos y normaliza espacios sin perder párrafos."""
    cleaned = text.replace("\x00", "")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        raise ValueError("El documento está vacío.")
    return cleaned


def ensure_file_exists(path: Path) -> None:
    """Verifica que el documento fuente exista."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {path}")


def validate_question(question: str) -> str:
    """Devuelve una pregunta limpia o rechaza una entrada vacía."""
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("La pregunta no puede estar vacía.")
    return cleaned


def validate_response_schema(response: dict[str, Any]) -> None:
    """Comprueba el contrato público exacto de la respuesta RAG."""
    if set(response) != EXPECTED_RESPONSE_KEYS:
        raise ValueError(
            "La respuesta debe contener exactamente user_question, "
            "system_answer y chunks_related."
        )
    if not isinstance(response["user_question"], str):
        raise TypeError("user_question debe ser texto.")
    if not isinstance(response["system_answer"], str):
        raise TypeError("system_answer debe ser texto.")
    if not isinstance(response["chunks_related"], list):
        raise TypeError("chunks_related debe ser una lista.")


def build_response(question: str, answer: str, documents: list[Any]) -> dict[str, Any]:
    """Crea la única forma pública de salida del chatbot."""
    response = {
        "user_question": question,
        "system_answer": answer,
        "chunks_related": [
            {
                "chunk_id": document.metadata["chunk_id"],
                "content": document.page_content,
            }
            for document in documents
        ],
    }
    validate_response_schema(response)
    return response
