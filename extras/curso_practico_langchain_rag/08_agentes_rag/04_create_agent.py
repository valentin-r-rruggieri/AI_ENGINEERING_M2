"""create_agent coordina modelo y herramientas.
Un agente agrega decisión y costo: para RAG simple se prefiere una cadena lineal.
# GUÍA DOCENTE
# CUÁNDO USAR: tareas con decisión real entre varias herramientas.
# DIFERENCIA: un agente decide pasos; un RAG lineal es más barato, predecible y
# suele bastar para preguntas sobre documentos.
# EN CLASE: preguntar primero '¿realmente necesitamos un agente aquí?'.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool

@tool
def buscar_faq(pregunta: str) -> str:
    """Busca evidencia en una FAQ pequeña."""
    return "Evidencia: el soporte se contacta por email."

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    try:
        from langchain.agents import create_agent
        from langchain_openai import ChatOpenAI
        agente = create_agent(ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0), [buscar_faq], system_prompt="Usa buscar_faq antes de responder.")
        resultado = agente.invoke({"messages": [{"role": "user", "content": "¿Cómo contacto soporte?"}]})
        print(resultado["messages"][-1].content)
    except ImportError:
        print("Instala el paquete langchain para usar create_agent.")
