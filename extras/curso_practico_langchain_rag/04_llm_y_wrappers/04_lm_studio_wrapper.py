"""LM Studio: usa ChatOpenAI contra un servidor local compatible.
Primero inicia el servidor local de LM Studio y carga un modelo de chat.
"""
import os
from langchain_openai import ChatOpenAI

modelo = ChatOpenAI(model=os.getenv("LM_STUDIO_MODEL", "local-model"), api_key="lm-studio", base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"), temperature=0, timeout=20)
try:
    print(modelo.invoke("Di hola en español.").content)
except Exception:
    print("Inicia el servidor local de LM Studio para ejecutar este ejemplo.")
