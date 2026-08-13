"""LCEL conecta componentes con |.
RunnablePassthrough deja pasar la salida del prompt para inspeccionarla sin código auxiliar.
# GUÍA DOCENTE
# CUÁNDO USAR: para conectar componentes LangChain de forma legible con |.
# DIFERENCIA: LCEL describe un pipeline declarativo; llamar todo manualmente es
# útil para debug, pero se vuelve difícil de mantener al crecer.
# EN CLASE: leer la cadena de izquierda a derecha como un flujo de datos.
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

prompt = PromptTemplate.from_template("Pregunta recibida: {question}")
cadena = prompt | RunnablePassthrough()

print(cadena.invoke({"question": "¿Qué es RAG?"}).to_string())
