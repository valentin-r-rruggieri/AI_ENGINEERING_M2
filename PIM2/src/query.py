"""Consulta el índice Chroma y devuelve la respuesta RAG pública."""
from __future__ import annotations

import argparse
import json
import sys

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import CHROMA_DIR, Settings, load_settings, require_openai_api_key
from utils import build_response, validate_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consulta el chatbot RAG FAQ de PeopleFlow.")
    parser.add_argument("--question", required=True, help="Pregunta del usuario.")
    return parser.parse_args()


def load_vector_store(settings: Settings) -> Chroma:
    """Abre la colección persistida y rechaza un índice inexistente."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError("No existe el índice. Ejecutá primero: python src/build_index.py")
    store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=OpenAIEmbeddings(model=settings.embedding_model),
        persist_directory=str(CHROMA_DIR),
    )
    if store._collection.count() == 0:
        raise FileNotFoundError("El índice está vacío. Ejecutá primero: python src/build_index.py")
    return store


def retrieve_documents(vector_store: Chroma, question: str, top_k: int):
    """Recupera entre dos y cinco fragments semánticamente relevantes."""
    if not 2 <= top_k <= 5:
        raise ValueError("TOP_K debe estar entre 2 y 5.")
    documents = vector_store.as_retriever(search_kwargs={"k": top_k}).invoke(question)
    if len(documents) < 2:
        raise RuntimeError("El índice no contiene suficientes chunks para responder.")
    return documents


def build_context(documents) -> str:
    """Convierte los chunks recuperados en contexto identificable para el LLM."""
    return "\n\n".join(
        f"[Chunk {document.metadata['chunk_id']}]\n"
        f"Fuente: {document.metadata.get('source', 'desconocida')}\n"
        f"Contenido:\n{document.page_content}"
        for document in documents
    )


def generate_answer(question: str, context: str, settings: Settings) -> str:
    """Genera una respuesta respaldada exclusivamente por el contexto."""
    prompt = ChatPromptTemplate.from_template(
        """Sos un asistente de soporte de PeopleFlow, una plataforma de RR.HH.

Respondé solamente con la información presente en el contexto recuperado.
No inventes datos ni uses conocimiento externo. Si el contexto no permite
responder de forma suficiente, indicá explícitamente: \"No tengo información
suficiente en la documentación disponible.\" Respondé de manera clara y breve.

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA:"""
    )
    chain = prompt | ChatOpenAI(model=settings.chat_model, temperature=0) | StrOutputParser()
    return chain.invoke({"context": context, "question": question}).strip()


def main() -> None:
    """Ejecuta una consulta y escribe solo el JSON público en stdout."""
    try:
        settings = load_settings()
        args = parse_args()
        question = validate_question(args.question)
        require_openai_api_key()
        documents = retrieve_documents(load_vector_store(settings), question, settings.top_k)
        response = build_response(question, generate_answer(question, build_context(documents), settings), documents)
    except (EnvironmentError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error de consulta: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
