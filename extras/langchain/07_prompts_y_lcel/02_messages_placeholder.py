# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""MessagesPlaceholder inserta historial como mensajes, no como texto plano.
Limita o resume historial para no consumir todo el contexto del modelo.
# GUÍA DOCENTE
# CUÁNDO USAR: conversaciones con preguntas de seguimiento.
# DIFERENCIA: historial ayuda a interpretar pronombres; no reemplaza retrieval
# ni debe tratarse como evidencia de la base documental.
# EN CLASE: quitar history y comparar si la pregunta sigue teniendo contexto.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.messages import HumanMessage, AIMessage
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([("system", "Eres un asistente RAG."), MessagesPlaceholder("history"), ("human", "{question}")])
mensajes = prompt.format_messages(history=[HumanMessage("Hablamos del plan Pro."), AIMessage("El plan Pro incluye API.")], question="¿Y cuántos usuarios permite?")

for mensaje in mensajes:
    print(mensaje.type, ":", mensaje.content)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
