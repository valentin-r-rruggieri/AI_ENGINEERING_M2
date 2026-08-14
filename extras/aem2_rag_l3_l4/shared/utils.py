# Este modulo reune helpers pequenos para que cada ejercicio se enfoque en una sola idea de RAG.
# Incluye impresion de Documents, busqueda local didactica, validacion de API key, medicion de
# tiempo y un proveedor de embeddings simple. No contiene una aplicacion: solo piezas reutilizables.

# annotations permite usar tipos modernos sin evaluarlos inmediatamente.
from __future__ import annotations

# os lee configuracion y claves desde las variables de entorno.
import os
# re tokeniza texto con una expresion regular sencilla.
import re
# sys permite agregar la raiz del curso al path de imports.
import sys
# time mide latencias de las etapas del pipeline.
import time
# Counter cuenta terminos para los embeddings locales didacticos.
from collections import Counter
# contextmanager convierte una funcion con yield en un bloque with reutilizable.
from contextlib import contextmanager
# Path construye rutas hacia .env, datos e indices.
from pathlib import Path
# Iterable describe colecciones de Documents que reciben los helpers.
from typing import Iterable

# load_dotenv carga OPENAI_API_KEY desde un archivo .env local.
from dotenv import load_dotenv
# Document representa evidencia textual junto con metadata de LangChain.
from langchain_core.documents import Document
# Embeddings define los metodos que debe tener cualquier proveedor de vectores.
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parents[1]


def prepare_imports() -> Path:
    """Permite ejecutar cualquier archivo directamente desde su propia carpeta."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    load_dotenv(ROOT / ".env")
    return ROOT


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def lexical_score(query: str, document: Document) -> float:
    query_tokens, doc_tokens = set(tokens(query)), set(tokens(document.page_content))
    return len(query_tokens & doc_tokens) / max(1, len(query_tokens))


def lexical_search(query: str, documents: Iterable[Document], k: int = 4) -> list[Document]:
    ranked = []
    for document in documents:
        copy = Document(page_content=document.page_content, metadata=dict(document.metadata))
        copy.metadata["score"] = round(lexical_score(query, copy), 3)
        ranked.append(copy)
    return sorted(ranked, key=lambda item: item.metadata["score"], reverse=True)[:k]


def show_documents(documents: Iterable[Document]) -> None:
    for index, document in enumerate(documents, start=1):
        print(f"[{index}] id={document.metadata.get('id', '?')} score={document.metadata.get('score', '-')}")
        print(f"    {document.page_content}")
        print(f"    metadata={document.metadata}")


def needs_openai() -> bool:
    if os.getenv("OPENAI_API_KEY"):
        return True
    print("OPENAI_API_KEY no esta configurada. Copia .env.example a .env y agrega tu clave para ejecutar este ejercicio cloud.")
    print("El flujo queda preparado; las credenciales se mantienen fuera del codigo.")
    return False


def optional_import(module: str, package: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        print(f"Falta la dependencia opcional '{package}'. Instalala con: pip install {package}")
        return False


class KeywordEmbeddings(Embeddings):
    """Embeddings locales deterministas para practicar la API de LangChain sin red."""

    vocabulary = ("vacaciones remoto horario contrasena salud reportes empleados dias trabajar casa jornada exportar soporte cobertura administradores").split()

    def _embed(self, text: str) -> list[float]:
        counts = Counter(tokens(text))
        return [float(counts[word]) for word in self.vocabulary]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@contextmanager
def elapsed(label: str):
    start = time.perf_counter()
    yield
    print(f"{label}: {(time.perf_counter() - start) * 1000:.2f} ms")


def percentile(values: list[float], percent: float) -> float:
    values = sorted(values)
    index = max(0, min(len(values) - 1, round((percent / 100) * (len(values) - 1))))
    return values[index]


def conclusion(text: str) -> None:
    print("\nCONCLUSION:")
    print(text)


# Resumen final: estos helpers simplifican los ejercicios, pero no reemplazan las piezas de LangChain.
# Cada script sigue mostrando explicitamente la carga, indexacion, retrieval o evaluacion que enseña.
