"""MessagesPlaceholder inserta historial como mensajes, no como texto plano.
Limita o resume historial para no consumir todo el contexto del modelo.
# GUÍA DOCENTE
# CUÁNDO USAR: conversaciones con preguntas de seguimiento.
# DIFERENCIA: historial ayuda a interpretar pronombres; no reemplaza retrieval
# ni debe tratarse como evidencia de la base documental.
# EN CLASE: quitar history y comparar si la pregunta sigue teniendo contexto.
"""
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([("system", "Eres un asistente RAG."), MessagesPlaceholder("history"), ("human", "{question}")])
mensajes = prompt.format_messages(history=[HumanMessage("Hablamos del plan Pro."), AIMessage("El plan Pro incluye API.")], question="¿Y cuántos usuarios permite?")

for mensaje in mensajes:
    print(mensaje.type, ":", mensaje.content)
