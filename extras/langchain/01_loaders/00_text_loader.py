# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""TextLoader: carga un .txt y lo transforma en Document.
El contenido queda en page_content y el origen en metadata['source'].
# GUÍA DOCENTE
# CUÁNDO USAR: para .txt UTF-8, FAQs, transcripciones o documentos planos.
# DIFERENCIA: TextLoader lee un archivo completo; DirectoryLoader repite esa idea
# sobre muchos archivos; un loader de PDF conserva páginas.
# EN CLASE: abrir metadata['source'] y distinguirla del texto de page_content.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.document_loaders import TextLoader

curso = Path(__file__).resolve().parents[1]
documentos = TextLoader(str(curso / "data" / "faq_empresa_saas.txt"), encoding="utf-8").load()

print("Cantidad de documentos:", len(documentos))
print("Metadata:", documentos[0].metadata)
print(documentos[0].page_content[:300])

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
