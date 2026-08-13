"""RAG con historial: el historial aclara la nueva pregunta, no es evidencia.
MessagesPlaceholder conserva roles para que el modelo interprete el seguimiento.
"""
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([("system", "Reformula la última pregunta para buscarla en una base documental."), MessagesPlaceholder("history"), ("human", "{question}")])
mensajes = prompt.format_messages(history=[HumanMessage("Háblame del plan Pro."), AIMessage("El plan Pro incluye API.")], question="¿Y cuántos usuarios permite?")

for mensaje in mensajes:
    print(mensaje.type, ":", mensaje.content)
