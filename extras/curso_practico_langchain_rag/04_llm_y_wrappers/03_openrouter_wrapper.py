"""OpenRouter: usa ChatOpenAI con base_url compatible.
Cambian URL, clave y nombre de modelo; el patrón de mensajes sigue igual.
# GUÍA DOCENTE
# CUÁNDO USAR: para acceder a modelos publicados mediante una API compatible.
# DIFERENCIA: el patrón LangChain es el mismo que OpenAI; cambian base_url,
# API key y nombre de modelo. Disponibilidad y precio dependen del proveedor.
# EN CLASE: comparar configuración, no claves ni resultados de modelos privados.
"""
import os
from langchain_openai import ChatOpenAI

if not os.getenv("OPENROUTER_API_KEY"):
    print("Configura OPENROUTER_API_KEY para este ejemplo opcional.")
else:
    modelo = ChatOpenAI(model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"), api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", temperature=0)
    print(modelo.invoke("Di hola en español.").content)
