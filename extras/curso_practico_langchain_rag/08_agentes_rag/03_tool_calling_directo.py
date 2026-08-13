"""bind_tools permite que ChatOpenAI solicite una herramienta.
Este paso solo muestra la solicitud: la aplicación decide cuándo y cómo ejecutarla.
# GUÍA DOCENTE
# CUÁNDO USAR: para inspeccionar qué tool pide el modelo antes de ejecutarla.
# DIFERENCIA: bind_tools solo permite solicitar una llamada; la aplicación debe
# validar permisos y ejecutar la herramienta por separado.
# EN CLASE: nunca usar tool_calls como autorización para acciones destructivas.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def consultar_ticket(ticket_id: str) -> str:
    """Consulta el estado de un ticket."""
    return f"Ticket {ticket_id}: abierto."

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    respuesta = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0).bind_tools([consultar_ticket]).invoke("Consulta el ticket T-42.")
    print(respuesta.tool_calls)
