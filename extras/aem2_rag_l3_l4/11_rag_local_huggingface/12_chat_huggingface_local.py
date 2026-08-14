# Este archivo muestra la variante de chat para un modelo local: Transformers se adapta primero
# como LLM y despues como ChatHuggingFace para recibir mensajes con roles. Requiere el stack local
# y descarga el modelo inicial la primera vez. Al ejecutarlo responde a una pregunta con sistema y usuario.
# sys permite importar el helper de dependencias desde la raiz del curso.
import sys
# Path obtiene esa raiz a partir de la ubicacion de este script.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# HumanMessage y SystemMessage representan mensajes con roles dentro de LangChain.
from langchain_core.messages import HumanMessage, SystemMessage

# optional_import evita cargar el modelo cuando todavia falta el stack local.
from shared.utils import optional_import

hf_ready = optional_import("langchain_huggingface", "langchain-huggingface")
transformers_ready = optional_import("transformers", "transformers torch accelerate")

if hf_ready and transformers_ready:
    # pipeline construye el generador de Transformers que corre en esta maquina.
    from transformers import pipeline
    # HuggingFacePipeline lo adapta como LLM y ChatHuggingFace lo adapta como chat model.
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    generator = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=80)
    chat = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=generator))

    # Los mensajes separan la instruccion de sistema de la pregunta del usuario.
    response = chat.invoke([
        SystemMessage(content="Responde en espanol y en una frase corta."),
        HumanMessage(content="Que es una base vectorial en RAG?"),
    ])
    print(response.content)
    print("ChatHuggingFace permite usar mensajes de LangChain sobre un pipeline local de Transformers.")

# Resumen final: ChatHuggingFace agrega roles de sistema y usuario sobre un modelo local de Transformers.
# Esto permite conservar la misma forma de conversar que con chat models cloud, pero usando recursos propios.
