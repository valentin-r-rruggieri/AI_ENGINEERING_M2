"""Un RAG debe tener una salida explícita cuando retrieval no encuentra evidencia.
No tener documentos recuperados es una señal para no completar la respuesta con invenciones.
"""
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([("system", "Responde solo con contexto. Si está vacío, responde: No tengo evidencia en los documentos."), ("human", "Contexto: {context}\nPregunta: {question}")])
mensajes = prompt.format_messages(context="", question="¿Cuál será el precio en 2035?")

for mensaje in mensajes:
    print(mensaje.type, ":", mensaje.content)
