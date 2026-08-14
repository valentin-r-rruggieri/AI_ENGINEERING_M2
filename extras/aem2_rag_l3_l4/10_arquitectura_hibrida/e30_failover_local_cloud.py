# Este archivo simula un RAG hibrido donde la ruta cloud falla y una ruta local toma la consulta.
# Muestra que el fallback mantiene disponibilidad, pero tambien debe registrarse porque podria
# recuperar evidencia diferente. Al ejecutarlo se ve el error simulado y los resultados locales.
# os lee si existe una configuracion cloud en las variables del entorno.
import os
# sys permite importar el corpus y helpers desde un script ejecutado solo.
import sys
# Path encuentra la raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS es la base de evidencia disponible para el fallback local.
from shared.dataset import DOCUMENTS
# lexical_search implementa el fallback y show_documents muestra los resultados usados.
from shared.utils import lexical_search, show_documents


def cloud_search(question):
    # Esta funcion simula una dependencia cloud que no esta configurada o sufrio un timeout.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Proveedor cloud no configurado")
    raise RuntimeError("Simulacion: timeout del proveedor cloud")


# Si la ruta principal falla, el fallback local mantiene la disponibilidad de la consulta.
try:
    results, route = cloud_search("Puedo trabajar desde casa?"), "cloud"
except RuntimeError as error:
    print("Fallo primario:", error)
    results, route = lexical_search("Puedo trabajar desde casa?", DOCUMENTS, k=2), "local"

print("Ruta usada:", route)
show_documents(results)
print("El failover debe ser observable porque las rutas pueden tener distinta calidad de retrieval.")

# Resumen final: un fallback local evita que una falla cloud deje sin respuesta a toda la aplicacion.
# La ruta usada y su calidad deben registrarse para saber cuando el sistema esta degradado.
