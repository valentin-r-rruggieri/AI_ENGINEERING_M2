# Este modulo contiene el corpus comun de LlamaIndex. Reutilizar los mismos Documents permite
# comparar carga, nodes, indices y retrievers sin cambiar el problema de negocio.

# Document combina texto con metadata dentro del modelo de datos de LlamaIndex.
from llama_index.core import Document

DOCUMENTS = [
    Document(text="Los empleados tienen 15 dias de vacaciones por ano. Las vacaciones deben solicitarse con al menos cinco dias de anticipacion.", metadata={"id": "vacaciones", "category": "rrhh", "source": "rrhh.md"}),
    Document(text="La empresa permite trabajar remoto hasta tres dias por semana.", metadata={"id": "remoto", "category": "rrhh", "source": "rrhh.md"}),
    Document(text="El horario laboral habitual es de 9 a 18 horas.", metadata={"id": "horario", "category": "rrhh", "source": "rrhh.md"}),
    Document(text="Los usuarios pueden restablecer su contrasena desde la pantalla de inicio de sesion.", metadata={"id": "password", "category": "soporte", "source": "soporte.md"}),
]

EVALUATION_QUERIES = [
    {"query": "Cuantos dias de vacaciones tengo?", "relevant_ids": {"vacaciones"}},
    {"query": "Puedo trabajar desde casa?", "relevant_ids": {"remoto"}},
    {"query": "Como recupero mi contrasena?", "relevant_ids": {"password"}},
]

# Resumen final: este corpus pequeno permite repetir los ejemplos y evaluar retrieval con IDs.
