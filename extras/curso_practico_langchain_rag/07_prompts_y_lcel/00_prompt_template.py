"""PromptTemplate inserta variables en una instrucción reutilizable.
Imprime el prompt final para auditar qué texto verá el modelo.
# GUÍA DOCENTE
# CUÁNDO USAR: una instrucción de texto con variables explícitas.
# DIFERENCIA: PromptTemplate produce una cadena; ChatPromptTemplate produce
# mensajes con roles. El segundo es más natural para chat models.
# EN CLASE: imprimir el prompt antes de llamar al modelo y detectar variables faltantes.
"""
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Responde solo con este contexto:\n{context}\n\nPregunta: {question}")
texto = prompt.format(context="El plan Pro incluye API.", question="¿Qué incluye Pro?")

print(texto)
