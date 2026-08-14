# Este archivo evalua retrieval con consultas cuya evidencia correcta ya esta etiquetada. Calcula
# Recall@K, Precision@K y MRR para que se pueda mejorar la busqueda antes de tocar el prompt o
# el LLM. Al ejecutarlo se ven rankings por pregunta y el promedio de las tres metricas.
# sys permite importar los datos de evaluacion sin instalar el proyecto.
import sys
# Path calcula la raiz del curso desde la ubicacion del ejercicio.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS es el corpus y EVALUATION_QUERIES contiene la evidencia esperada por pregunta.
from shared.dataset import DOCUMENTS, EVALUATION_QUERIES
# lexical_search produce los rankings que se comparan con las etiquetas.
from shared.utils import lexical_search

# Estas listas acumulan resultados por consulta para evaluar la recuperacion, no la redaccion del LLM.
recalls, precisions, reciprocal_ranks = [], [], []
for item in EVALUATION_QUERIES:
    ids = [doc.metadata["id"] for doc in lexical_search(item["query"], DOCUMENTS, k=3)]
    hits = set(ids) & item["relevant_ids"]
    recalls.append(len(hits) / len(item["relevant_ids"]))
    precisions.append(len(hits) / len(ids))

    # MRR premia que el primer documento relevante aparezca arriba en el ranking.
    rank = next((index + 1 for index, ident in enumerate(ids) if ident in item["relevant_ids"]), None)
    reciprocal_ranks.append(1 / rank if rank else 0)
    print(item["query"], "->", ids)

print(f"Recall@3={sum(recalls)/len(recalls):.2f} Precision@3={sum(precisions)/len(precisions):.2f} MRR={sum(reciprocal_ranks)/len(reciprocal_ranks):.2f}")
print("Recall mide cobertura, precision mide ruido y MRR mide que tan temprano aparece la evidencia.")

# Resumen final: evaluar retrieval separa calidad de evidencia y calidad de redaccion del LLM.
# Recall, precision y MRR muestran fallas distintas y orientan que parte del indice mejorar.
