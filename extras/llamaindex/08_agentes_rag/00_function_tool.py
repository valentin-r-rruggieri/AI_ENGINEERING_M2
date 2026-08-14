# Este archivo presenta FunctionTool, la forma de convertir una funcion Python en una herramienta
# que un agente de LlamaIndex puede invocar. No crea un agente todavia: primero muestra el contrato
# de entrada y salida de la tool. Al ejecutarlo llama la herramienta de forma directa.

# FunctionTool adapta una funcion comun al sistema de herramientas de LlamaIndex.
from llama_index.core.tools import FunctionTool

def buscar_horario() -> str:
    return "El horario laboral habitual es de 9 a 18 horas."

tool = FunctionTool.from_defaults(fn=buscar_horario, name="buscar_horario")
print(tool.call())

# Resumen final: una tool encapsula una accion verificable antes de delegarla a un agente.
