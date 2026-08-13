"""MessagesPlaceholder inserta historial como mensajes, no como texto plano.
Limita o resume historial para no consumir todo el contexto del modelo.
"""
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([("system", "Eres un asistente RAG."), MessagesPlaceholder("history"), ("human", "{question}")])
mensajes = prompt.format_messages(history=[HumanMessage("Hablamos del plan Pro."), AIMessage("El plan Pro incluye API.")], question="¿Y cuántos usuarios permite?")

for mensaje in mensajes:
    print(mensaje.type, ":", mensaje.content)
