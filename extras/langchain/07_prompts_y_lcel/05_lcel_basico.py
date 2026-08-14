# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""LCEL conecta componentes con |.
RunnablePassthrough deja pasar la salida del prompt para inspeccionarla sin código auxiliar.
# GUÍA DOCENTE
# CUÁNDO USAR: para conectar componentes LangChain de forma legible con |.
# DIFERENCIA: LCEL describe un pipeline declarativo; llamar todo manualmente es
# útil para debug, pero se vuelve difícil de mantener al crecer.
# EN CLASE: leer la cadena de izquierda a derecha como un flujo de datos.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.prompts import PromptTemplate
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.runnables import RunnablePassthrough

prompt = PromptTemplate.from_template("Pregunta recibida: {question}")
cadena = prompt | RunnablePassthrough()

print(cadena.invoke({"question": "¿Qué es RAG?"}).to_string())

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
