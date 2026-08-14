# Este archivo usa PromptTemplate de LlamaIndex para preparar una instruccion con contexto y
# pregunta. No llama un modelo: deja visible el texto que recibira el LLM. Es util para revisar
# grounding y reglas de abstencion antes de conectar un QueryEngine real.

# PromptTemplate interpolar variables dentro de instrucciones reutilizables.
from llama_index.core import PromptTemplate

template = PromptTemplate("Responde solo con este contexto: {context}\nPregunta: {question}\nRespuesta:")
print(template.format(context="Vacaciones: 15 dias.", question="Cuantos dias tengo?"))

# Resumen final: un prompt claro delimita que datos son contexto y que texto es la pregunta.
