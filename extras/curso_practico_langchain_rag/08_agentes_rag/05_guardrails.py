"""Las herramientas se diseñan con alcance mínimo y de solo lectura.
El agente puede sugerir una llamada; la aplicación sigue aplicando permisos y límites.
# GUÍA DOCENTE
# CUÁNDO USAR: siempre que un modelo pueda llamar herramientas.
# DIFERENCIA: el prompt indica comportamiento; los guardrails lo hacen cumplir
# mediante tools de alcance limitado, permisos y validación de argumentos.
# EN CLASE: clasificar una tool como solo lectura, escritura o prohibida.
"""
from langchain_core.tools import tool

@tool
def buscar_documento(consulta: str) -> str:
    """Busca solamente documentación pública. No escribe ni elimina datos."""
    return f"Resultado público para: {consulta}"

print(buscar_documento.name)
print(buscar_documento.description)
print(buscar_documento.invoke({"consulta": "licencias"}))
