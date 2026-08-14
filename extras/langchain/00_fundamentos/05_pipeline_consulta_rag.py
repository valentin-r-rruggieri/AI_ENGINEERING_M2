# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""Fundamentos: consulta RAG autónoma con evidencia, prompt y OpenAI opcional.

Uso local, sin LLM:
    python 05_pipeline_consulta_rag.py --question "Como solicito vacaciones?"

Uso con OpenAI, solo si OPENAI_API_KEY existe en .env:
    python 05_pipeline_consulta_rag.py --question "Como solicito vacaciones?" --use-openai

El índice local usa FakeEmbeddings para que el ejemplo siempre pueda ejecutarse.
Como esos vectores no tienen semántica real, el script complementa retrieval con una
selección léxica determinista. OpenAI opcional recibe únicamente el contexto elegido.
# GUÍA DOCENTE
# CUÁNDO USAR: para explicar el pipeline online de RAG.
# FLUJO: pregunta -> retrieval -> contexto con fuentes -> prompt -> respuesta.
# DIFERENCIA: sin --use-openai enseña retrieval y prompt sin costo; con OpenAI
# agrega generación grounded.
# EN CLASE: preguntar algo sin evidencia y verificar que no invente.
"""

# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
import argparse
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from datetime import date
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from hashlib import sha256
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
import os
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
import re

# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from dotenv import load_dotenv
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.document_loaders import TextLoader
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.embeddings import FakeEmbeddings
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.vectorstores import FAISS
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.documents import Document
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.prompts import ChatPromptTemplate
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_text_splitters import RecursiveCharacterTextSplitter


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    while current.parent != current:
        if (current / ".env.example").is_file() and (current / "data").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("No se encontró la raíz del curso.")


def stable_id(source: str, chunk_number: int, content: str) -> str:
    return sha256(f"{source}|{chunk_number}|{content}".encode("utf-8")).hexdigest()[:16]


def build_or_open_index(root: Path) -> tuple[FAISS, list[Document]]:
    """Siempre deja un índice disponible y devuelve también chunks para ranking léxico."""
    source = root / "data" / "faq_empresa_saas.txt"
    index_path = root / "storage" / "fundamentos_faiss"
    embeddings = FakeEmbeddings(size=64)

    documents = TextLoader(str(source), encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50).split_documents(documents)
    ids: list[str] = []
    for number, chunk in enumerate(chunks):
        chunk_id = stable_id(source.name, number, chunk.page_content)
        chunk.metadata.update(
            {
                "chunk_id": chunk_id,
                "chunk_number": number,
                "source": source.name,
                "indexed_on": date.today().isoformat(),
            }
        )
        ids.append(chunk_id)

    if index_path.exists():
        store = FAISS.load_local(
            str(index_path), embeddings, allow_dangerous_deserialization=True
        )
    else:
        store = FAISS.from_documents(chunks, embeddings, ids=ids)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        store.save_local(str(index_path))
    return store, chunks


STOPWORDS = {
    "a", "al", "ante", "como", "con", "de", "del", "el", "en", "es", "esta",
    "este", "la", "las", "lo", "los", "mi", "o", "para", "por", "que", "se",
    "su", "una", "un", "y", "ya",
}


def tokens(text: str) -> set[str]:
    """Extrae términos útiles para un baseline léxico, descartando palabras funcionales."""
    return {
        token
        for token in re.findall(r"[a-záéíóúñ]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def lexical_retrieve(question: str, chunks: list[Document], k: int = 3) -> list[Document]:
    """Ranking determinista para que el ejemplo local no dependa de vectores falsos."""
    question_tokens = tokens(question)
    scored = []
    for chunk in chunks:
        overlap = len(question_tokens & tokens(chunk.page_content))
        scored.append((overlap, -chunk.metadata["chunk_number"], chunk))
    scored.sort(reverse=True)
    return [chunk for score, _, chunk in scored if score > 0][:k]


def format_context(chunks: list[Document]) -> str:
    return "\n\n".join(
        f"[Fuente: {chunk.metadata['source']} | chunk: {chunk.metadata['chunk_number']}]\n"
        f"{chunk.page_content}"
        for chunk in chunks
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consulta RAG de fundamentos.")
    parser.add_argument("--question", required=True, help="Pregunta para el FAQ.")
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Genera una respuesta real si OPENAI_API_KEY está configurada.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question = args.question.strip()
    if not question:
        raise ValueError("La pregunta no puede estar vacía.")

    root = find_project_root(Path.cwd())
    load_dotenv(root / ".env", override=False)
    _, chunks = build_or_open_index(root)
    retrieved = lexical_retrieve(question, chunks)
    context = format_context(retrieved)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Respondé solamente con el contexto. Si no alcanza, respondé: "
                "No hay información suficiente en el contexto recuperado.",
            ),
            ("human", "CONTEXTO:\n{context}\n\nPREGUNTA:\n{question}"),
        ]
    )
    prompt_value = prompt.invoke({"context": context or "(sin evidencia)", "question": question})

    print("PREGUNTA:", question)
    print("\nCHUNKS RECUPERADOS:", len(retrieved))
    for chunk in retrieved:
        print(
            f"- chunk {chunk.metadata['chunk_number']}: "
            f"{chunk.page_content[:170].replace(chr(10), ' ')}"
        )
    print("\nPROMPT FINAL:")
    print(prompt_value.to_string())

    if not args.use_openai:
        print("\nMODO LOCAL: no se llamó a un LLM. Se mostró evidencia y prompt.")
        return
    if not os.getenv("OPENAI_API_KEY", "").strip():
        print("\nOPENAI NO CONFIGURADO: no se generó respuesta.")
        return

    # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)
    answer = model.invoke(prompt_value).content
    print("\nRESPUESTA GROUNDED:")
    print(answer)
    print("\nFUENTES:")
    for chunk in retrieved:
        print(f"- {chunk.metadata['source']} / chunk {chunk.metadata['chunk_number']}")


if __name__ == "__main__":
    main()

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
