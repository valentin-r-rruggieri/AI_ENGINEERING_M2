"""StructuredTool expone una herramienta de varios parámetros.
La validación de tipos reduce llamadas ambiguas antes de ejecutar acciones reales.
# GUÍA DOCENTE
# CUÁNDO USAR: herramientas con varios parámetros o validaciones.
# DIFERENCIA: @tool infiere parámetros simples; StructuredTool permite declarar
# un esquema Pydantic explícito con rangos, descripciones y valores por defecto.
# EN CLASE: enviar un límite inválido y explicar la validación.
"""
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class EntradaBusqueda(BaseModel):
    consulta: str = Field(description="Texto a buscar")
    limite: int = Field(default=2, ge=1, le=5)

def buscar(consulta: str, limite: int = 2) -> str:
    return f"Búsqueda: {consulta}. Límite: {limite}."

herramienta = StructuredTool.from_function(buscar, name="buscar_documentos", description="Busca documentos.", args_schema=EntradaBusqueda)
print(herramienta.invoke({"consulta": "licencias", "limite": 2}))
