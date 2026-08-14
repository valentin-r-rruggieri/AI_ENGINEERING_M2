# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""Las herramientas se diseñan con alcance mínimo y de solo lectura.
El agente puede sugerir una llamada; la aplicación sigue aplicando permisos y límites.
# GUÍA DOCENTE
# CUÁNDO USAR: siempre que un modelo pueda llamar herramientas.
# DIFERENCIA: el prompt indica comportamiento; los guardrails lo hacen cumplir
# mediante tools de alcance limitado, permisos y validación de argumentos.
# EN CLASE: clasificar una tool como solo lectura, escritura o prohibida.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.tools import tool

@tool
def buscar_documento(consulta: str) -> str:
    """Busca solamente documentación pública. No escribe ni elimina datos."""
    return f"Resultado público para: {consulta}"

print(buscar_documento.name)
print(buscar_documento.description)
print(buscar_documento.invoke({"consulta": "licencias"}))

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
