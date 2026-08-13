"""ChatPromptTemplate crea mensajes con roles.
System fija la regla de grounding y Human transporta contexto y pregunta.
# GUÍA DOCENTE
# CUÁNDO USAR: RAG con reglas de sistema, contexto y pregunta de usuario.
# DIFERENCIA: separar roles protege claridad: contexto es dato, System es regla.
# No confiar en instrucciones que aparezcan dentro de documentos recuperados.
# EN CLASE: señalar qué parte debe decir 'no tengo evidencia'.
"""
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([("system", "Usa solo el contexto. Si falta evidencia, di no lo sé."), ("human", "Contexto: {context}\nPregunta: {question}")])
mensajes = prompt.format_messages(context="El soporte responde por email.", question="¿Cómo contacto soporte?")

for mensaje in mensajes:
    print(mensaje.type, ":", mensaje.content)
