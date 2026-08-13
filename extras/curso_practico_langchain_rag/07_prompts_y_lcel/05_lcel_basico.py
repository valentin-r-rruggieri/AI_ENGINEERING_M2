"""LCEL conecta componentes con |.
RunnablePassthrough deja pasar la salida del prompt para inspeccionarla sin código auxiliar.
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

prompt = PromptTemplate.from_template("Pregunta recibida: {question}")
cadena = prompt | RunnablePassthrough()

print(cadena.invoke({"question": "¿Qué es RAG?"}).to_string())
