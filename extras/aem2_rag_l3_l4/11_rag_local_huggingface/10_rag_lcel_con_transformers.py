# Este archivo construye el RAG local completo: embeddings Hugging Face, FAISS, retriever, prompt,
# modelo Transformers y parser conectados mediante LCEL. Requiere todas las dependencias locales
# y descarga modelos en la primera ejecucion. Al ejecutarlo responde usando solo el contexto local.
# sys agrega shared al path para ejecutar el script sin instalar el proyecto.
import sys
# Path localiza la raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# PromptTemplate arma un prompt de texto para un LLM local no conversacional.
from langchain_core.prompts import PromptTemplate
# StrOutputParser devuelve solo el texto producido por el modelo.
from langchain_core.output_parsers import StrOutputParser
# RunnablePassthrough conserva la pregunta mientras la otra rama recupera contexto.
from langchain_core.runnables import RunnablePassthrough

# DOCUMENTS es el corpus que FAISS indexa de forma local.
from shared.dataset import DOCUMENTS
# optional_import verifica el stack local antes de crear el pipeline completo.
from shared.utils import optional_import

faiss_ready = optional_import("faiss", "faiss-cpu")
hf_ready = optional_import("langchain_huggingface", "langchain-huggingface sentence-transformers")
transformers_ready = optional_import("transformers", "transformers torch accelerate")

if faiss_ready and hf_ready and transformers_ready:
    # FAISS sera el indice vectorial del retriever local.
    from langchain_community.vectorstores import FAISS
    # HuggingFaceEmbeddings vectoriza y HuggingFacePipeline adapta Transformers a LangChain.
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
    # pipeline carga el modelo generativo local desde Transformers.
    from transformers import pipeline

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    retriever = FAISS.from_documents(DOCUMENTS, embeddings).as_retriever(search_kwargs={"k": 2})

    # Este formatter conserva IDs para que el modelo pueda mencionar la fuente elegida.
    format_docs = lambda docs: "\n".join(f"[{doc.metadata['id']}] {doc.page_content}" for doc in docs)
    prompt = PromptTemplate.from_template(
        "Responde solamente con el contexto. Si falta evidencia, responde no tengo evidencia.\n\nContexto:\n{context}\n\nPregunta: {question}\nRespuesta:"
    )
    llm = HuggingFacePipeline(pipeline=pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=90))

    # LCEL expresa el RAG como una cadena visible de retrieval, prompt, LLM y salida.
    chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    print(chain.invoke("Cuantos dias de vacaciones tengo?"))

# Resumen final: LCEL hace visible la cadena local completa desde retrieval hasta texto final.
# La calidad depende de chunks, embeddings, indice, prompt y modelo; no solo del ultimo componente.
