# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""RAG local mínimo: loader → splitter → embeddings → vectorstore → retrieval.
Primero se inspecciona el contexto recuperado; un LLM solo se agrega después.
# GUÍA DOCENTE
# CUÁNDO USAR: primera demostración RAG sin API ni generación.
# FLUJO: documento -> splitter -> FakeEmbeddings -> FAISS -> chunks recuperados.
# DIFERENCIA: demuestra retrieval; un RAG con LLM agrega una respuesta sobre ese
# contexto. Primero validar evidencia evita culpar al modelo por un mal retrieval.
# EN CLASE: leer chunks recuperados antes de hablar de prompts.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.document_loaders import TextLoader
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.embeddings import FakeEmbeddings
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.vectorstores import FAISS

curso = Path(__file__).resolve().parents[1]
documentos = TextLoader(str(curso / "data" / "faq_empresa_saas.txt"), encoding="utf-8").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50).split_documents(documentos)
store = FAISS.from_documents(chunks, FakeEmbeddings(size=32))
resultados = store.similarity_search("¿Cómo contacto soporte?", k=3)

print("Chunks creados:", len(chunks))
for documento in resultados:
    print("-----")
    print(documento.page_content)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
