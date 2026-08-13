"""Fundamentos: pipeline real de indexación local con FAISS.

Uso:
    python 04_pipeline_indexacion_local.py

Flujo ejecutado:
    TXT -> TextLoader -> RecursiveCharacterTextSplitter -> metadata/IDs
    -> FakeEmbeddings -> FAISS -> almacenamiento local

FakeEmbeddings permite validar el contrato técnico sin gastar API. No mide
relevancia semántica real: ese punto se cubre con OpenAIEmbeddings más adelante.
# GUÍA DOCENTE
# CUÁNDO USAR: para explicar el pipeline offline de RAG completo.
# FLUJO: loader -> chunks -> metadata/IDs -> embeddings -> FAISS persistente.
# DIFERENCIA: indexar prepara conocimiento; todavía no responde preguntas.
# EN CLASE: variar chunk_size y comparar cantidad/tamaño de chunks.
"""

from datetime import date
from hashlib import sha256
from pathlib import Path
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    while current.parent != current:
        if (current / ".env.example").is_file() and (current / "data").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("No se encontró la raíz del curso.")


def stable_id(source: str, chunk_number: int, content: str) -> str:
    raw = f"{source}|{chunk_number}|{content}".encode("utf-8")
    return sha256(raw).hexdigest()[:16]


def build_index(root: Path, replace: bool = True) -> tuple[FAISS, list[str]]:
    """Carga, divide e indexa el FAQ; devuelve store e IDs de chunks."""
    source = root / "data" / "faq_empresa_saas.txt"
    documents = TextLoader(str(source), encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    ids: list[str] = []
    for number, chunk in enumerate(chunks):
        chunk_id = stable_id(source.name, number, chunk.page_content)
        chunk.metadata.update(
            {
                "chunk_id": chunk_id,
                "chunk_number": number,
                "source": source.name,
                "index_version": "fundamentos-v1",
                "indexed_on": date.today().isoformat(),
            }
        )
        ids.append(chunk_id)

    embeddings = FakeEmbeddings(size=64)
    store = FAISS.from_documents(chunks, embeddings, ids=ids)
    index_path = root / "storage" / "fundamentos_faiss"
    if replace and index_path.exists():
        shutil.rmtree(index_path)
    store.save_local(str(index_path))
    return store, ids


def main() -> None:
    root = find_project_root(Path.cwd())
    store, ids = build_index(root)
    index_path = root / "storage" / "fundamentos_faiss"

    # Verificación de persistencia: se vuelve a abrir la estructura guardada.
    reloaded = FAISS.load_local(
        str(index_path), FakeEmbeddings(size=64), allow_dangerous_deserialization=True
    )
    sample = reloaded.get_by_ids([ids[0]])

    assert len(ids) >= 4, "El FAQ debería producir varios chunks."
    assert len(sample) == 1, "No se pudo recuperar el primer chunk persistido."
    assert sample[0].metadata["chunk_id"] == ids[0]

    print("DOCUMENTOS CARGADOS: 1")
    print("CHUNKS GENERADOS:", len(ids))
    print("RUTA DEL ÍNDICE:", index_path)
    print("\nPRIMER CHUNK:")
    print(sample[0].page_content[:280])
    print("\nMETADATA:", sample[0].metadata)
    print("\nOK: indexación y reapertura de FAISS completadas.")


if __name__ == "__main__":
    main()
