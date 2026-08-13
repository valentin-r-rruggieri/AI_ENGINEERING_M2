"""CSVLoader: cada fila del CSV se convierte en un Document.
Es útil para tickets, productos y tablas: conserva número de fila en metadata.
"""
from pathlib import Path
from langchain_community.document_loaders import CSVLoader

curso = Path(__file__).resolve().parents[1]
documentos = CSVLoader(str(curso / "data" / "ejemplo_productos.csv"), encoding="utf-8").load()

print("Filas cargadas:", len(documentos))
print(documentos[0].page_content)
print(documentos[0].metadata)
