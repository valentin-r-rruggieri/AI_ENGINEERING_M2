# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""PromptTemplate inserta variables en una instrucción reutilizable.
Imprime el prompt final para auditar qué texto verá el modelo.
# GUÍA DOCENTE
# CUÁNDO USAR: una instrucción de texto con variables explícitas.
# DIFERENCIA: PromptTemplate produce una cadena; ChatPromptTemplate produce
# mensajes con roles. El segundo es más natural para chat models.
# EN CLASE: imprimir el prompt antes de llamar al modelo y detectar variables faltantes.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Responde solo con este contexto:\n{context}\n\nPregunta: {question}")
texto = prompt.format(context="El plan Pro incluye API.", question="¿Qué incluye Pro?")

print(texto)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
