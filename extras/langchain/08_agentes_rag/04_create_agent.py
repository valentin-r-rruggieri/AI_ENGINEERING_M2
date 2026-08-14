# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""create_agent coordina modelo y herramientas.
Un agente agrega decisión y costo: para RAG simple se prefiere una cadena lineal.
# GUÍA DOCENTE
# CUÁNDO USAR: tareas con decisión real entre varias herramientas.
# DIFERENCIA: un agente decide pasos; un RAG lineal es más barato, predecible y
# suele bastar para preguntas sobre documentos.
# EN CLASE: preguntar primero '¿realmente necesitamos un agente aquí?'.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
import os
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from dotenv import load_dotenv
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
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
        # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
        from langchain.agents import create_agent
        # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
        from langchain_openai import ChatOpenAI
        agente = create_agent(ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0), [buscar_faq], system_prompt="Usa buscar_faq antes de responder.")
        resultado = agente.invoke({"messages": [{"role": "user", "content": "¿Cómo contacto soporte?"}]})
        print(resultado["messages"][-1].content)
    except ImportError:
        print("Instala el paquete langchain para usar create_agent.")

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
