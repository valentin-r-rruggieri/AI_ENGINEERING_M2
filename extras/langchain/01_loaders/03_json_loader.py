# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""JSONLoader: selecciona registros de JSON mediante jq_schema.
jq_schema define qué parte del JSON es un documento; content_key define el texto.
Requiere el extra opcional: pip install jq
# GUÍA DOCENTE
# CUÁNDO USAR: APIs, catálogos y registros JSON con estructura conocida.
# DIFERENCIA: jq_schema elige registros; content_key elige el campo que será
# texto. A diferencia de CSV, JSON puede tener objetos anidados.
# EN CLASE: modificar jq_schema y explicar por qué una mala selección indexa ruido.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.document_loaders import JSONLoader

curso = Path(__file__).resolve().parents[1]

try:
    documentos = JSONLoader(str(curso / "data" / "ejemplo_faq.json"), jq_schema=".faqs[]", content_key="respuesta").load()
    print(documentos[0].page_content)
    print(documentos[0].metadata)
except ImportError:
    print("Instala jq para ejecutar JSONLoader: pip install jq")

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
