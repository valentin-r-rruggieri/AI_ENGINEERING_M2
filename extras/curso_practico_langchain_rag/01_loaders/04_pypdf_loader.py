"""PyPDFLoader: carga un PDF digital página por página.
metadata['page'] permite citar la página. Un PDF escaneado necesita OCR externo.
"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

archivo = Path(__file__).resolve().parents[3] / "RAG_Chunking_Engineering.pdf"

if archivo.exists():
    documentos = PyPDFLoader(str(archivo)).load()
    print("Páginas:", len(documentos))
    print("Página:", documentos[0].metadata["page"] + 1)
    print(documentos[0].page_content[:300])
else:
    print("Coloca un PDF junto al curso para ejecutar este ejemplo.")
