"""Evaluador bonus para respuestas generadas por el RAG."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import Settings, load_settings, require_openai_api_key


def validate_evaluation(evaluation: dict[str, Any]) -> None:
    """Verifica el contrato independiente de la evaluación bonus."""
    if set(evaluation) != {"score", "reason"}:
        raise ValueError("La evaluación debe contener exactamente score y reason.")
    score = evaluation["score"]
    reason = evaluation["reason"]
    if not isinstance(score, int) or isinstance(score, bool):
        raise TypeError("score debe ser un entero.")
    if not 0 <= score <= 10:
        raise ValueError("score debe estar entre 0 y 10.")
    if not isinstance(reason, str) or len(reason.strip()) < 50:
        raise ValueError("reason debe contener al menos 50 caracteres.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa una respuesta del chatbot RAG.")
    parser.add_argument("--question", required=True, help="Pregunta original.")
    parser.add_argument("--answer", required=True, help="Respuesta a evaluar.")
    parser.add_argument(
        "--chunks-file",
        required=True,
        help="JSON con chunks_related o con una respuesta RAG completa.",
    )
    return parser.parse_args()


def load_chunks(path: str) -> list[dict[str, Any]]:
    """Lee los chunks desde una lista o desde una respuesta pública guardada."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    chunks = payload["chunks_related"] if isinstance(payload, dict) else payload
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("El archivo debe contener una lista no vacía de chunks.")
    return chunks


def evaluate_response(
    question: str, answer: str, chunks: list[dict[str, Any]], settings: Settings
) -> dict[str, Any]:
    """Solicita una evaluación JSON centrada en evidencia y fidelidad."""
    prompt = ChatPromptTemplate.from_template(
        """Evaluá esta respuesta de un sistema RAG. Considerá la relevancia de los
chunks, el grounding (ausencia de información inventada) y la completitud.
Devolvé exclusivamente JSON válido con score entero de 0 a 10 y reason en
español de al menos 50 caracteres con observaciones concretas.

Pregunta: {question}
Respuesta: {answer}
Chunks: {chunks}"""
    )
    schema = {
        "name": "rag_evaluation",
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "reason": {"type": "string", "minLength": 50},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
    }
    structured_llm = ChatOpenAI(model=settings.chat_model, temperature=0).with_structured_output(schema)
    evaluation = (prompt | structured_llm).invoke(
        {"question": question, "answer": answer, "chunks": json.dumps(chunks, ensure_ascii=False)}
    )
    validate_evaluation(evaluation)
    return evaluation


def main() -> None:
    """Evalúa una respuesta y emite solo el JSON de evaluación."""
    try:
        args = parse_args()
        settings = load_settings()
        require_openai_api_key()
        evaluation = evaluate_response(
            args.question.strip(), args.answer.strip(), load_chunks(args.chunks_file), settings
        )
    except (EnvironmentError, FileNotFoundError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"Error de evaluación: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
