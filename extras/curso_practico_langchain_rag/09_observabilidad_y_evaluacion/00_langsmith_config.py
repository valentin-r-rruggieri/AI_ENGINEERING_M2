"""LangSmith es opcional y registra trazas de ejecuciones LangChain.
Solo se comprueba si las variables existen; nunca se imprimen secretos.
# GUÍA DOCENTE
# CUÁNDO USAR: al depurar cadenas, comparar experimentos o medir latencia.
# DIFERENCIA: LangSmith observa ejecuciones; no mejora por sí solo retrieval
# ni la respuesta. Su uso es opcional y debe respetar privacidad.
# EN CLASE: explicar qué datos viajan a una traza y qué no se debe enviar.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
print("LANGSMITH_TRACING:", "configurada" if os.getenv("LANGSMITH_TRACING") else "ausente")
print("LANGSMITH_API_KEY:", "configurada" if os.getenv("LANGSMITH_API_KEY") else "ausente")
print("LANGSMITH_PROJECT:", os.getenv("LANGSMITH_PROJECT", "sin proyecto"))
