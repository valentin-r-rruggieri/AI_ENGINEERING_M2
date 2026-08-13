"""ChatPromptTemplate crea mensajes con roles.
System fija la regla de grounding y Human transporta contexto y pregunta.
"""
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([("system", "Usa solo el contexto. Si falta evidencia, di no lo sé."), ("human", "Contexto: {context}\nPregunta: {question}")])
mensajes = prompt.format_messages(context="El soporte responde por email.", question="¿Cómo contacto soporte?")

for mensaje in mensajes:
    print(mensaje.type, ":", mensaje.content)
