# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""OpenRouter: usa ChatOpenAI con base_url compatible.
Cambian URL, clave y nombre de modelo; el patrón de mensajes sigue igual.
# GUÍA DOCENTE
# CUÁNDO USAR: para acceder a modelos publicados mediante una API compatible.
# DIFERENCIA: el patrón LangChain es el mismo que OpenAI; cambian base_url,
# API key y nombre de modelo. Disponibilidad y precio dependen del proveedor.
# EN CLASE: comparar configuración, no claves ni resultados de modelos privados.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
import os
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_openai import ChatOpenAI

if not os.getenv("OPENROUTER_API_KEY"):
    print("Configura OPENROUTER_API_KEY para este ejemplo opcional.")
else:
    modelo = ChatOpenAI(model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"), api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", temperature=0)
    print(modelo.invoke("Di hola en español.").content)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
