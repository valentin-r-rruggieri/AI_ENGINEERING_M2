# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""RecursiveCharacterTextSplitter: opción inicial para prosa.
Prueba párrafos, líneas, palabras y caracteres para respetar chunk_size.
# GUÍA DOCENTE
# CUÁNDO USAR: primera opción para texto general, FAQs y documentación plana.
# DIFERENCIA: intenta párrafo, línea, espacio y carácter para respetar chunk_size;
# no entiende tokens ni estructura semántica profunda.
# EN CLASE: comparar overlap 0 vs 50 y buscar información partida entre chunks.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.document_loaders import TextLoader
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_text_splitters import RecursiveCharacterTextSplitter

curso = Path(__file__).resolve().parents[1]
documentos = TextLoader(str(curso / "data" / "faq_empresa_saas.txt"), encoding="utf-8").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50).split_documents(documentos)

print("Chunks:", len(chunks))
print(chunks[0].page_content)
print(chunks[0].metadata)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
