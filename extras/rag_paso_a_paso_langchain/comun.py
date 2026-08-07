"""Utilidades compartidas por los pasos: rutas, embeddings y LLM de LangChain.

No es un "paso" en si mismo: es la caja de herramientas que los pasos siguientes
van a ir importando y reusando.
"""
import json
from pathlib import Path
import sys

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.append(str(PROJECT_ROOT / "extras" / "embeddings" / "data"))

from documentos import DOCUMENTOS, PREGUNTA  # noqa: E402

ESTADO = Path(__file__).resolve().parent / "estado"
ESTADO.mkdir(exist_ok=True)

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

# A que departamento pertenece cada fuente, para poder practicar filtros mas adelante.
DEPARTAMENTO_POR_FUENTE = {
    "manual_red.md": "IT",
    "facturacion.md": "Finance",
    "seguridad.md": "Security",
}


def guardar_documentos(documentos: list[Document], nombre: str) -> None:
    """Guarda una lista de Document usando la serializacion propia de Document (model_dump)."""
    datos = [documento.model_dump() for documento in documentos]
    with open(ESTADO / nombre, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)


def cargar_documentos(nombre: str) -> list[Document]:
    """Reconstruye la lista de Document guardada por guardar_documentos()."""
    with open(ESTADO / nombre, encoding="utf-8") as f:
        datos = json.load(f)
    return [Document(**dato) for dato in datos]
