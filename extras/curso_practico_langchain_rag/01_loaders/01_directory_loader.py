"""DirectoryLoader: carga todos los archivos de una carpeta.
Elegir glob evita indexar formatos que no corresponden al loader elegido.
"""
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader

curso = Path(__file__).resolve().parents[1]
documentos = DirectoryLoader(
    str(curso / "data"),
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
).load()

print("Archivos cargados:", len(documentos))
for documento in documentos:
    print(documento.metadata["source"], "|", len(documento.page_content), "caracteres")
