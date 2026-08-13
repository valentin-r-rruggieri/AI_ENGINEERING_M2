"""with_structured_output: solicita una respuesta validada contra un esquema.
El esquema da forma a la salida; no prueba que la información sea verdadera.
# GUÍA DOCENTE
# CUÁNDO USAR: cuando la aplicación necesita campos, no texto libre.
# DIFERENCIA: con_structured_output valida forma contra Pydantic; no garantiza que
# el contenido sea cierto. Para RAG todavía se debe pasar contexto grounded.
# EN CLASE: agregar un campo obligatorio y observar la estructura resultante.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class Concepto(BaseModel):
    termino: str = Field(description="Nombre del concepto")
    definicion: str = Field(description="Definición corta")

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    respuesta = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0).with_structured_output(Concepto).invoke("Explica el concepto de chunk en RAG.")
    print(respuesta.model_dump())
