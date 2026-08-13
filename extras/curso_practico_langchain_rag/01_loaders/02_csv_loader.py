"""CSVLoader: cada fila del CSV se convierte en un Document.
Es útil para tickets, productos y tablas: conserva número de fila en metadata.
# GUÍA DOCENTE
# CUÁNDO USAR: para filas independientes como productos, tickets o inventarios.
# DIFERENCIA: cada fila se vuelve Document; para JSON anidado conviene JSONLoader.
# LÍMITE: una tabla relacional compleja puede requerir consultas SQL, no RAG.
# EN CLASE: ver cómo nombre de columna y valor se convierten en texto recuperable.
"""
from pathlib import Path
from langchain_community.document_loaders import CSVLoader

curso = Path(__file__).resolve().parents[1]
documentos = CSVLoader(str(curso / "data" / "ejemplo_productos.csv"), encoding="utf-8").load()

print("Filas cargadas:", len(documentos))
print(documentos[0].page_content)
print(documentos[0].metadata)
