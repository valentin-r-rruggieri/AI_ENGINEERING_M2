# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""RAG con historial: el historial aclara la nueva pregunta, no es evidencia.
MessagesPlaceholder conserva roles para que el modelo interprete el seguimiento.
# GUÍA DOCENTE
# CUÁNDO USAR: chat con seguimientos como '¿y cuántos usuarios permite?'.
# DIFERENCIA: historial resuelve referencias conversacionales; el corpus sigue
# siendo la fuente de verdad y no se reemplaza por mensajes previos.
# EN CLASE: comparar una pregunta con y sin historial.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.messages import HumanMessage, AIMessage
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([("system", "Reformula la última pregunta para buscarla en una base documental."), MessagesPlaceholder("history"), ("human", "{question}")])
mensajes = prompt.format_messages(history=[HumanMessage("Háblame del plan Pro."), AIMessage("El plan Pro incluye API.")], question="¿Y cuántos usuarios permite?")

for mensaje in mensajes:
    print(mensaje.type, ":", mensaje.content)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
