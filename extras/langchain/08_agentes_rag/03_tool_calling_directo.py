# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""bind_tools permite que ChatOpenAI solicite una herramienta.
Este paso solo muestra la solicitud: la aplicación decide cuándo y cómo ejecutarla.
# GUÍA DOCENTE
# CUÁNDO USAR: para inspeccionar qué tool pide el modelo antes de ejecutarla.
# DIFERENCIA: bind_tools solo permite solicitar una llamada; la aplicación debe
# validar permisos y ejecutar la herramienta por separado.
# EN CLASE: nunca usar tool_calls como autorización para acciones destructivas.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
import os
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from dotenv import load_dotenv
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.tools import tool
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
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

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
