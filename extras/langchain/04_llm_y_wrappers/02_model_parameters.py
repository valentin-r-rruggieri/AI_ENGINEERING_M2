# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""Parámetros de ChatOpenAI: temperatura, límite de salida y timeout.
Se ajustan según variación aceptada, presupuesto y latencia de la aplicación.
# GUÍA DOCENTE
# CUÁNDO USAR: al ajustar calidad, costo y tiempo de respuesta de una app.
# DIFERENCIA: temperature controla variación; max_tokens limita la salida;
# timeout define cuánto esperar antes de fallar.
# EN CLASE: cambiar un parámetro por vez para no confundir sus efectos.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
import os
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from dotenv import load_dotenv
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    modelo = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0, max_tokens=60, timeout=30)
    print(modelo.invoke("Explica chunking en una frase.").content)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
