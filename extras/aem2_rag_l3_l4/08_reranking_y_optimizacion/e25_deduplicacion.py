# Este archivo muestra por que se debe eliminar contenido repetido antes de indexar. Duplica un
# Document del corpus, deduplica mediante su ID y compara cantidades antes y despues. Al ejecutarlo
# se entiende como evitar costo inutil y contexto redundante dentro de un RAG.
# sys permite encontrar el corpus compartido desde una ejecucion directa.
import sys
# Path resuelve la raiz del curso a partir de este archivo.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS contiene IDs estables que permiten demostrar la deduplicacion.
from shared.dataset import DOCUMENTS

# Indexar dos veces el mismo documento aumenta costo y puede repetir evidencia dentro del contexto.
dirty = DOCUMENTS + [DOCUMENTS[0]]

# En este corpus el ID de metadata es una identidad estable para decidir que es un duplicado.
unique = {document.metadata["id"]: document for document in dirty}
print("Antes:", len(dirty), "documentos")
print("Despues:", len(unique), "documentos")
print("IDs:", list(unique))
print("La clave de deduplicacion debe representar el significado de duplicado del negocio.")

# Resumen final: deduplicar antes de indexar evita costo extra y contexto repetido en las respuestas.
# La identidad puede ser un ID de origen o un hash de contenido segun el tipo de corpus.
