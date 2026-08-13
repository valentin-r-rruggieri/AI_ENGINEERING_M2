"""@tool convierte una acción en una herramienta que un modelo puede llamar.
El nombre, tipos y docstring son el contrato visible para el agente.
# GUÍA DOCENTE
# CUÁNDO USAR: una capacidad pequeña y segura que un agente puede decidir usar.
# DIFERENCIA: una función normal no expone esquema; @tool crea nombre, descripción
# y argumentos legibles para el modelo.
# EN CLASE: mostrar el schema y explicar por qué docstring y tipos importan.
"""
from langchain_core.tools import tool

@tool
def buscar_politica(tema: str) -> str:
    """Busca una política interna por tema."""
    return "Soporte: se contacta por email." if tema == "soporte" else "No hay una política para ese tema."

print(buscar_politica.invoke({"tema": "soporte"}))
print(buscar_politica.args_schema.model_json_schema())
