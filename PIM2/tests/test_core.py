from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from evaluator import validate_evaluation
from query import retrieve_documents
from utils import build_response, clean_text, validate_question, validate_response_schema


def test_clean_text_normalizes_and_rejects_empty() -> None:
    assert clean_text(" A\t\n\n\nB ") == "A\n\nB"
    with pytest.raises(ValueError):
        clean_text(" \n\t ")


def test_question_cannot_be_empty() -> None:
    assert validate_question("  Consulta  ") == "Consulta"
    with pytest.raises(ValueError):
        validate_question("   ")


def test_public_response_has_exactly_three_keys() -> None:
    document = SimpleNamespace(metadata={"chunk_id": 1}, page_content="Contenido")
    response = build_response("Pregunta", "Respuesta", [document])
    assert set(response) == {"user_question", "system_answer", "chunks_related"}
    validate_response_schema(response)


def test_public_response_rejects_extra_key() -> None:
    with pytest.raises(ValueError):
        validate_response_schema(
            {"user_question": "P", "system_answer": "R", "chunks_related": [], "score": 9}
        )


def test_top_k_must_be_between_two_and_five() -> None:
    with pytest.raises(ValueError):
        retrieve_documents(None, "Pregunta", 1)
    with pytest.raises(ValueError):
        retrieve_documents(None, "Pregunta", 6)


def test_evaluation_contract() -> None:
    valid = {
        "score": 8,
        "reason": "La evidencia recuperada responde la pregunta y la respuesta no agrega políticas ajenas al contexto disponible.",
    }
    validate_evaluation(valid)
    with pytest.raises(ValueError):
        validate_evaluation({"score": 11, "reason": valid["reason"]})
