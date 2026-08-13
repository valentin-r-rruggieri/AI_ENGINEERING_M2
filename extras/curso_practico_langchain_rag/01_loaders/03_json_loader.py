"""JSONLoader: selecciona registros de JSON mediante jq_schema.
jq_schema define qué parte del JSON es un documento; content_key define el texto.
Requiere el extra opcional: pip install jq
# GUÍA DOCENTE
# CUÁNDO USAR: APIs, catálogos y registros JSON con estructura conocida.
# DIFERENCIA: jq_schema elige registros; content_key elige el campo que será
# texto. A diferencia de CSV, JSON puede tener objetos anidados.
# EN CLASE: modificar jq_schema y explicar por qué una mala selección indexa ruido.
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
