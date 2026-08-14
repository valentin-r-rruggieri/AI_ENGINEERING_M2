# Este archivo muestra la integracion OpenAI de LlamaIndex para generacion. Requiere clave y el
# paquete opcional, por eso se protege la importacion. Al ejecutarlo configurado genera una respuesta
# corta; sin configuracion explica exactamente que dependencia falta.

# os lee OPENAI_API_KEY sin dejarla escrita en el codigo.
import os
# importlib.util comprueba si el wrapper de OpenAI esta instalado.
import importlib.util

if importlib.util.find_spec("llama_index.llms.openai") and os.getenv("OPENAI_API_KEY"):
    # OpenAI adapta el chat model al contrato LLM de LlamaIndex.
    from llama_index.llms.openai import OpenAI
    print(OpenAI(model="gpt-4o-mini").complete("Explica RAG en una frase.").text)
else:
    print("Configura OPENAI_API_KEY e instala llama-index-llms-openai para este ejemplo.")

# Resumen final: el LLM se agrega al final del pipeline, despues de recuperar evidencia.
