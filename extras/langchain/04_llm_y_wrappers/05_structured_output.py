# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""with_structured_output: solicita una respuesta validada contra un esquema.
El esquema da forma a la salida; no prueba que la información sea verdadera.
# GUÍA DOCENTE
# CUÁNDO USAR: cuando la aplicación necesita campos, no texto libre.
# DIFERENCIA: con_structured_output valida forma contra Pydantic; no garantiza que
# el contenido sea cierto. Para RAG todavía se debe pasar contexto grounded.
# EN CLASE: agregar un campo obligatorio y observar la estructura resultante.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
import os
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from dotenv import load_dotenv
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pydantic import BaseModel, Field
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
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

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
