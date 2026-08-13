"""with_structured_output: solicita una respuesta validada contra un esquema.
El esquema da forma a la salida; no prueba que la información sea verdadera.
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
