"""E05: indexar, persistir, recargar y evaluar Recall@K."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import deterministic_embedding, read_json, split_words, top_k, write_json

root=Path(__file__).parent
chunks=split_words((root/"data"/"corpus.txt").read_text(encoding="utf-8"),25,5,"corpus")
vectors=[deterministic_embedding(c.content) for c in chunks]
hits=0; cases=read_json(root/"data"/"queries.json")
for case in cases:
    # Recall@3 cuenta una consulta como acierto si la evidencia aparece en Top-3.
    result=top_k(deterministic_embedding(case["question"]),chunks,vectors,3)
    hits+=int(any(word in " ".join(x["chunk"]["content"].lower() for x in result) for word in case["keywords"]))
report={"recall_at_3":hits/len(cases),"queries":len(cases)}
# El reporte queda guardado para comparar cambios de índice o parámetros más adelante.
write_json(root/"data"/"benchmark.json",report); print(report)
