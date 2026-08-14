# Este archivo une todas las piezas cloud de un RAG: embeddings OpenAI, Chroma, retriever,
# prompt, ChatOpenAI y parser de salida dentro de una cadena LCEL. Requiere OPENAI_API_KEY;
# al ejecutarlo con la clave se genera una respuesta basada en documentos y sus IDs citables.
# sys permite que el script importe dataset y utils sin instalar el curso como paquete.
import sys
# Path encuentra la raiz del curso a partir del archivo actual.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Chroma indexa el corpus y recupera los chunks semanticamente cercanos.
from langchain_chroma import Chroma
# StrOutputParser extrae texto plano de la respuesta del modelo.
from langchain_core.output_parsers import StrOutputParser
# ChatPromptTemplate crea el prompt con contexto y pregunta como variables.
from langchain_core.prompts import ChatPromptTemplate
# RunnablePassthrough conserva la pregunta mientras otra rama recupera contexto.
from langchain_core.runnables import RunnablePassthrough

# DOCUMENTS es el corpus que se indexa en Chroma.
from shared.dataset import DOCUMENTS
# needs_openai verifica la clave antes de importar y llamar al proveedor cloud.
from shared.utils import needs_openai

# El ejercicio solo intenta conectar al proveedor si existe una clave en .env.
if needs_openai():
    # ChatOpenAI genera la respuesta y OpenAIEmbeddings vectoriza corpus y pregunta.
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    # Chroma usa embeddings cloud para transformar corpus y pregunta al mismo espacio vectorial.
    retriever = Chroma.from_documents(DOCUMENTS, OpenAIEmbeddings(), collection_name="e19").as_retriever(search_kwargs={"k": 2})

    # El formatter conserva los IDs, que luego sirven para citar la evidencia elegida.
    format_docs = lambda docs: "\n".join(f"[{d.metadata['id']}] {d.page_content}" for d in docs)
    prompt = ChatPromptTemplate.from_template("Responde solo con el contexto y cita [id].\n{context}\nPregunta: {question}")

    # LCEL hace visible el flujo: pregunta -> retriever -> contexto -> prompt -> modelo -> texto.
    chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    print(chain.invoke("Cuantos dias de vacaciones tengo?"))

# Resumen final: una cadena RAG clara deja visibles retrieval, contexto, prompt y generacion.
# OpenAI genera al final; la respuesta depende primero de que Chroma recupere buena evidencia.
