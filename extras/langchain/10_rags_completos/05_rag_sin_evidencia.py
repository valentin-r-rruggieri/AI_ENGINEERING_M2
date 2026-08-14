# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""Un RAG debe tener una salida explícita cuando retrieval no encuentra evidencia.
No tener documentos recuperados es una señal para no completar la respuesta con invenciones.
# GUÍA DOCENTE
# CUÁNDO USAR: cualquier RAG que no debe alucinar.
# DIFERENCIA: una pregunta fuera del corpus no es un error técnico; requiere una
# política explícita de respuesta y, idealmente, threshold/evaluación.
# EN CLASE: usar preguntas de precios futuros o datos ausentes.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([("system", "Responde solo con contexto. Si está vacío, responde: No tengo evidencia en los documentos."), ("human", "Contexto: {context}\nPregunta: {question}")])
mensajes = prompt.format_messages(context="", question="¿Cuál será el precio en 2035?")

for mensaje in mensajes:
    print(mensaje.type, ":", mensaje.content)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
