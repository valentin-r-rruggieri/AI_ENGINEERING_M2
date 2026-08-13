"""Parámetros de ChatOpenAI: temperatura, límite de salida y timeout.
Se ajustan según variación aceptada, presupuesto y latencia de la aplicación.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    modelo = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0, max_tokens=60, timeout=30)
    print(modelo.invoke("Explica chunking en una frase.").content)
