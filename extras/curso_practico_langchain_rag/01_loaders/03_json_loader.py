"""JSONLoader: selecciona registros de JSON mediante jq_schema.
jq_schema define qué parte del JSON es un documento; content_key define el texto.
Requiere el extra opcional: pip install jq
"""
from pathlib import Path
from langchain_community.document_loaders import JSONLoader

curso = Path(__file__).resolve().parents[1]

try:
    documentos = JSONLoader(str(curso / "data" / "ejemplo_faq.json"), jq_schema=".faqs[]", content_key="respuesta").load()
    print(documentos[0].page_content)
    print(documentos[0].metadata)
except ImportError:
    print("Instala jq para ejecutar JSONLoader: pip install jq")
