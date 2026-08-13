"""ChatOpenAI: llama un modelo conversacional.
temperature=0 reduce la variación y es una configuración inicial útil para RAG.
# GUÍA DOCENTE
# CUÁNDO USAR: generar una respuesta una vez que ya existe contexto confiable.
# DIFERENCIA: ChatOpenAI genera lenguaje; OpenAIEmbeddings representa significado.
# En RAG, el modelo no busca documentos: recibe el contexto que retrieval encontró.
# EN CLASE: variar temperature entre 0 y 1 y comparar consistencia.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    respuesta = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0).invoke("Define RAG en una oración.")
    print(respuesta.content)
