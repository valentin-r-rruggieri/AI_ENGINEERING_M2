# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""LM Studio: usa ChatOpenAI contra un servidor local compatible.
Primero inicia el servidor local de LM Studio y carga un modelo de chat.
# GUÍA DOCENTE
# CUÁNDO USAR: demostraciones locales, privacidad o trabajo sin proveedor cloud.
# DIFERENCIA: LM Studio ejecuta un servidor en la propia máquina; OpenAI usa cloud.
# Un modelo local consume RAM/VRAM y su calidad depende del modelo descargado.
# EN CLASE: iniciar el server antes y comparar base_url local contra cloud.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
import os
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_openai import ChatOpenAI

modelo = ChatOpenAI(model=os.getenv("LM_STUDIO_MODEL", "local-model"), api_key="lm-studio", base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"), temperature=0, timeout=20)
try:
    print(modelo.invoke("Di hola en español.").content)
except Exception:
    print("Inicia el servidor local de LM Studio para ejecutar este ejemplo.")

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
