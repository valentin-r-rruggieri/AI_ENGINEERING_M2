"""PromptTemplate inserta variables en una instrucción reutilizable.
Imprime el prompt final para auditar qué texto verá el modelo.
"""
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Responde solo con este contexto:\n{context}\n\nPregunta: {question}")
texto = prompt.format(context="El plan Pro incluye API.", question="¿Qué incluye Pro?")

print(texto)
