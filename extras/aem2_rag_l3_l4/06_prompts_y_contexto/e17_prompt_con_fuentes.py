# Este archivo agrega fuentes al contexto y usa LCEL para conectar formateo y prompt. Muestra
# como conservar el ID y archivo de cada Document para pedir citas verificables al LLM. Al
# ejecutarlo se imprime el prompt con las fuentes ya preparadas para una respuesta RAG.
# sys permite usar los modulos shared al ejecutar este archivo solo.
import sys
# Path obtiene la raiz del curso desde la ruta actual.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ChatPromptTemplate define el mensaje con variables de contexto y pregunta.
from langchain_core.prompts import ChatPromptTemplate
# RunnableLambda transforma una funcion de Python en un paso de LCEL.
from langchain_core.runnables import RunnableLambda

# DOCUMENTS es la fuente de evidencia que se citara.
from shared.dataset import DOCUMENTS
# lexical_search obtiene los documentos relevantes antes de formatear las fuentes.
from shared.utils import lexical_search


def format_sources(question: str) -> dict[str, str]:
    # La fuente acompana al texto para que el modelo pueda devolver una cita verificable.
    docs = lexical_search(question, DOCUMENTS, k=2)
    context = "\n".join(f"[{doc.metadata['id']} | {doc.metadata['source']}] {doc.page_content}" for doc in docs)
    return {"question": question, "context": context}


# LCEL une una funcion simple con un prompt usando |. Cada paso sigue siendo invocable por separado.
prompt = ChatPromptTemplate.from_template("Responde usando el contexto y cita [id].\n{context}\n\nPregunta: {question}")
chain = RunnableLambda(format_sources) | prompt
print(chain.invoke("Como recupero mi contrasena?").to_string())
print("Las citas hacen que una respuesta RAG sea verificable y no solo convincente.")

# Resumen final: conservar IDs y fuentes convierte un contexto recuperado en evidencia citable.
# LCEL permite componer y probar el formateo de fuentes antes de conectar un modelo generativo.
