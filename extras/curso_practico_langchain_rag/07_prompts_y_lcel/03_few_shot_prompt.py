"""FewShotPromptTemplate agrega ejemplos de formato.
Úsalo cuando la instrucción sola no logra una respuesta consistente; los ejemplos consumen tokens.
# GUÍA DOCENTE
# CUÁNDO USAR: formato de salida repetible o comportamiento difícil de describir.
# DIFERENCIA: zero-shot solo instruye; few-shot además muestra ejemplos. Más
# ejemplos aumentan consistencia, pero también tokens/costo.
# EN CLASE: cambiar el ejemplo y observar cómo condiciona el formato final.
"""
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

ejemplo = PromptTemplate.from_template("Pregunta: {pregunta}\nRespuesta: {respuesta}")
prompt = FewShotPromptTemplate(
    examples=[{"pregunta": "¿Capital de Francia?", "respuesta": "París."}],
    example_prompt=ejemplo,
    suffix="Pregunta: {pregunta}\nRespuesta:",
    input_variables=["pregunta"],
)

print(prompt.format(pregunta="¿Capital de Argentina?"))
