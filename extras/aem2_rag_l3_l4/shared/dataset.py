# Este modulo no es un ejercicio aislado: concentra el corpus comun de todo el curso. Cada
# Document representa una politica interna con ID, fuente y categoria, y las consultas incluyen
# la evidencia esperada para poder medir retrieval. Los scripts importan estos datos sin duplicarlos.

# Document es la estructura LangChain que guarda evidencia textual y metadata.
from langchain_core.documents import Document

DOCUMENTS = [
    Document(page_content="Los empleados tienen 15 dias de vacaciones por ano. Las vacaciones deben solicitarse con al menos cinco dias de anticipacion.", metadata={"id": "vacaciones", "source": "rrhh.md", "category": "rrhh"}),
    Document(page_content="La empresa permite trabajar de forma remota hasta tres dias por semana.", metadata={"id": "remoto", "source": "rrhh.md", "category": "rrhh"}),
    Document(page_content="El horario laboral habitual es de 9 a 18 horas.", metadata={"id": "horario", "source": "rrhh.md", "category": "rrhh"}),
    Document(page_content="Los usuarios pueden restablecer su contrasena desde la pantalla de inicio de sesion.", metadata={"id": "password", "source": "soporte.md", "category": "soporte"}),
    Document(page_content="La empresa ofrece cobertura medica a todos los empleados.", metadata={"id": "salud", "source": "beneficios.md", "category": "beneficios"}),
    Document(page_content="Los administradores pueden exportar reportes desde el panel de analitica.", metadata={"id": "reportes", "source": "analytics.md", "category": "producto"}),
]

EVALUATION_QUERIES = [
    {"query": "Cuantos dias de vacaciones tengo?", "relevant_ids": {"vacaciones"}},
    {"query": "Puedo trabajar desde casa?", "relevant_ids": {"remoto"}},
    {"query": "Como recupero mi contrasena?", "relevant_ids": {"password"}},
    {"query": "A que hora termina la jornada laboral?", "relevant_ids": {"horario"}},
]

# Resumen final: un corpus pequeno, estable y etiquetado permite repetir todos los ejercicios.
# Las consultas con IDs relevantes convierten ejemplos didacticos en evaluaciones de retrieval.
