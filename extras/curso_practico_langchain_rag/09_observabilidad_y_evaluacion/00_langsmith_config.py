"""LangSmith es opcional y registra trazas de ejecuciones LangChain.
Solo se comprueba si las variables existen; nunca se imprimen secretos.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
print("LANGSMITH_TRACING:", "configurada" if os.getenv("LANGSMITH_TRACING") else "ausente")
print("LANGSMITH_API_KEY:", "configurada" if os.getenv("LANGSMITH_API_KEY") else "ausente")
print("LANGSMITH_PROJECT:", os.getenv("LANGSMITH_PROJECT", "sin proyecto"))
