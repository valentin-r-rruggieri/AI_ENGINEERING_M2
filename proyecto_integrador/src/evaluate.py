from __future__ import annotations

import json

from openai import OpenAI

from .config import Settings, api_key
from .models import EvaluationResult, RAGAnswer


def evaluate_answer(answer: RAGAnswer, settings: Settings | None = None) -> EvaluationResult:
    """Evaluador opcional: releva grounding y completitud sin alterar la respuesta pública."""
    settings = settings or Settings.from_env()
    evidence = "\n".join(
        f"[{chunk.chunk_id}] {chunk.content}" for chunk in answer.chunks_related
    )
    prompt = f"""Evaluá la respuesta RAG con evidencia disponible.
Pregunta: {answer.user_question}
Respuesta: {answer.system_answer}
Evidencia: {evidence}
Devolvé JSON estricto con score entero de 0 a 10 y reason de al menos 50 caracteres.
El score debe penalizar información que no esté respaldada por la evidencia."""
    response = OpenAI(api_key=api_key()).responses.create(
        model=settings.generation_model, input=prompt,
    )
    try:
        return EvaluationResult.model_validate(json.loads(response.output_text))
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("El evaluador no devolvió JSON válido.") from error

