"""OpenRouter: usa ChatOpenAI con base_url compatible.
Cambian URL, clave y nombre de modelo; el patrón de mensajes sigue igual.
"""
import os
from langchain_openai import ChatOpenAI

if not os.getenv("OPENROUTER_API_KEY"):
    print("Configura OPENROUTER_API_KEY para este ejemplo opcional.")
else:
    modelo = ChatOpenAI(model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"), api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", temperature=0)
    print(modelo.invoke("Di hola en español.").content)
