"""bind_tools permite que ChatOpenAI solicite una herramienta.
Este paso solo muestra la solicitud: la aplicación decide cuándo y cómo ejecutarla.
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
