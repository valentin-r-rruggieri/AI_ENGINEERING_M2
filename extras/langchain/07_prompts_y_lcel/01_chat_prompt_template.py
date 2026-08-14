# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""ChatPromptTemplate crea mensajes con roles.
System fija la regla de grounding y Human transporta contexto y pregunta.
# GUÍA DOCENTE
# CUÁNDO USAR: RAG con reglas de sistema, contexto y pregunta de usuario.
# DIFERENCIA: separar roles protege claridad: contexto es dato, System es regla.
# No confiar en instrucciones que aparezcan dentro de documentos recuperados.
# EN CLASE: señalar qué parte debe decir 'no tengo evidencia'.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([("system", "Usa solo el contexto. Si falta evidencia, di no lo sé."), ("human", "Contexto: {context}\nPregunta: {question}")])
mensajes = prompt.format_messages(context="El soporte responde por email.", question="¿Cómo contacto soporte?")

for mensaje in mensajes:
    print(mensaje.type, ":", mensaje.content)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
