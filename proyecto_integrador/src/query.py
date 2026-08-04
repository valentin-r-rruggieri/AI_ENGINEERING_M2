from __future__ import annotations

import argparse

from openai import OpenAI

from .config import Settings, api_key
from .embeddings import embed_texts
from .models import RAGAnswer
from .stores import create_store


def build_context(chunks: list[object]) -> str:
    return "\n\n".join(
        f"[{item.chunk_id}] {item.content}" for item in chunks
    )


def generate_answer(question: str, context: str, settings: Settings) -> str:
    prompt = f"""Respondé en español usando solamente el contexto.
No inventes información ni uses conocimiento externo. Si la evidencia no alcanza,
decí exactamente que no hay información suficiente en el contexto.

CONTEXTO:
{context}

PREGUNTA:
{question}"""
    response = OpenAI(api_key=api_key()).responses.create(
        model=settings.generation_model, input=prompt,
    )
    answer = response.output_text.strip()
    if not answer:
        raise RuntimeError("El modelo no devolvió texto.")
    return answer


def answer_question(question: str, backend: str = "chroma") -> RAGAnswer:
    if not question or not question.strip():
        raise ValueError("La pregunta no puede estar vacía.")
    settings = Settings.from_env()
    vector = embed_texts([question], settings)[0]
    chunks = create_store(backend, settings.storage_dir).search(vector, settings.top_k)
    if not 2 <= len(chunks) <= 5:
        raise RuntimeError("El retrieval debe devolver entre 2 y 5 chunks.")
    return RAGAnswer(
        user_question=question.strip(),
        system_answer=generate_answer(question, build_context(chunks), settings),
        chunks_related=chunks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta el FAQ RAG.")
    parser.add_argument("--backend", choices=["chroma", "faiss"], default="chroma")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    print(answer_question(args.question, args.backend).model_dump_json(indent=2))


if __name__ == "__main__":
    main()

