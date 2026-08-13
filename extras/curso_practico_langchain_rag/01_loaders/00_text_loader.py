"""TextLoader: carga un .txt y lo transforma en Document.
El contenido queda en page_content y el origen en metadata['source'].
"""
from pathlib import Path
from langchain_community.document_loaders import TextLoader

curso = Path(__file__).resolve().parents[1]
documentos = TextLoader(str(curso / "data" / "faq_empresa_saas.txt"), encoding="utf-8").load()

print("Cantidad de documentos:", len(documentos))
print("Metadata:", documentos[0].metadata)
print(documentos[0].page_content[:300])
