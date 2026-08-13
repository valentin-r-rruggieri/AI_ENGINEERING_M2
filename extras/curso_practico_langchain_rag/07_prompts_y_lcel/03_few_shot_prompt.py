"""FewShotPromptTemplate agrega ejemplos de formato.
Úsalo cuando la instrucción sola no logra una respuesta consistente; los ejemplos consumen tokens.
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
