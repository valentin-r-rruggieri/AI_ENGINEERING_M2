# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""Mensajes: System contiene reglas; Human contiene la pregunta.
Separar roles permite inspeccionar instrucciones y datos antes de llamar al modelo.
# GUÍA DOCENTE
# CUÁNDO USAR: cuando hay reglas permanentes y una pregunta de usuario.
# DIFERENCIA: System define comportamiento; Human trae solicitud o contexto.
# No mezclar contenido recuperado con instrucciones de sistema.
# EN CLASE: imprimir mensajes y preguntar qué rol corresponde a cada texto.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
import os
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from dotenv import load_dotenv
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.messages import SystemMessage, HumanMessage
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    mensajes = [SystemMessage("Responde en español y de forma breve."), HumanMessage("¿Qué es un embedding?")]
    respuesta = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0).invoke(mensajes)
    print(respuesta.content)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
