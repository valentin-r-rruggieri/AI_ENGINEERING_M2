# Este archivo ilustra dos formas seguras de probar un retriever nuevo: shadow traffic calcula
# una respuesta candidata sin exponerla, y canary la muestra solo a una fraccion de usuarios.
# Al ejecutarlo se comparan ambos rankings y se ve cual ruta responde al bucket elegido.
# sys habilita imports compartidos desde un archivo que se ejecuta por separado.
import sys
# Path localiza la raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS es el corpus de ambas rutas para aislar el cambio al retriever.
from shared.dataset import DOCUMENTS
# lexical_search genera el ranking de referencia que luego compara la ruta sombra.
from shared.utils import lexical_search

# La ruta sombra ejecuta una configuracion candidata sin afectar a todos los usuarios.
question, user_bucket = "Cuantos dias de vacaciones tengo?", 7
primary = lexical_search(question, DOCUMENTS, k=2)
shadow = list(reversed(primary))  # Simula que otro retriever produce un orden distinto.

# Un canary expone la candidata solo a una fraccion pequena de buckets.
route = "canary" if user_bucket < 10 else "primary"
chosen = shadow if route == "canary" else primary

print("Ruta que responde:", route)
print("Principal:", [doc.metadata["id"] for doc in primary])
print("Sombra:   ", [doc.metadata["id"] for doc in shadow])
print("Respuesta visible:", chosen[0].page_content)
print("Antes de promover una ruta se comparan latencia, recall y overlap con trafico real.")

# Resumen final: shadow permite medir una ruta nueva sin afectar usuarios y canary la expone poco a poco.
# Ambas tecnicas reducen riesgo al migrar embeddings, retrievers o configuraciones de un RAG.
