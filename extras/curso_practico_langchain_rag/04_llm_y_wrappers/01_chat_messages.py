"""Mensajes: System contiene reglas; Human contiene la pregunta.
Separar roles permite inspeccionar instrucciones y datos antes de llamar al modelo.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    mensajes = [SystemMessage("Responde en español y de forma breve."), HumanMessage("¿Qué es un embedding?")]
    respuesta = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0).invoke(mensajes)
    print(respuesta.content)
