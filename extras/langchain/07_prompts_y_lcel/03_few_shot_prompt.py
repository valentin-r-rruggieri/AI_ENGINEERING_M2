# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""FewShotPromptTemplate agrega ejemplos de formato.
Úsalo cuando la instrucción sola no logra una respuesta consistente; los ejemplos consumen tokens.
# GUÍA DOCENTE
# CUÁNDO USAR: formato de salida repetible o comportamiento difícil de describir.
# DIFERENCIA: zero-shot solo instruye; few-shot además muestra ejemplos. Más
# ejemplos aumentan consistencia, pero también tokens/costo.
# EN CLASE: cambiar el ejemplo y observar cómo condiciona el formato final.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

ejemplo = PromptTemplate.from_template("Pregunta: {pregunta}\nRespuesta: {respuesta}")
prompt = FewShotPromptTemplate(
    examples=[{"pregunta": "¿Capital de Francia?", "respuesta": "París."}],
    example_prompt=ejemplo,
    suffix="Pregunta: {pregunta}\nRespuesta:",
    input_variables=["pregunta"],
)

print(prompt.format(pregunta="¿Capital de Argentina?"))

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
