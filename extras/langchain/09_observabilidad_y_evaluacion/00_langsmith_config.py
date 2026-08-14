# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""LangSmith es opcional y registra trazas de ejecuciones LangChain.
Solo se comprueba si las variables existen; nunca se imprimen secretos.
# GUÍA DOCENTE
# CUÁNDO USAR: al depurar cadenas, comparar experimentos o medir latencia.
# DIFERENCIA: LangSmith observa ejecuciones; no mejora por sí solo retrieval
# ni la respuesta. Su uso es opcional y debe respetar privacidad.
# EN CLASE: explicar qué datos viajan a una traza y qué no se debe enviar.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
import os
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
print("LANGSMITH_TRACING:", "configurada" if os.getenv("LANGSMITH_TRACING") else "ausente")
print("LANGSMITH_API_KEY:", "configurada" if os.getenv("LANGSMITH_API_KEY") else "ausente")
print("LANGSMITH_PROJECT:", os.getenv("LANGSMITH_PROJECT", "sin proyecto"))

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
