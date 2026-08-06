"""E05: recuperador trazable con API real."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import embed_openai, split_words, top_k, write_json

root = Path(__file__).parent
# La misma estrategia de chunking se aplica al corpus antes de calcular embeddings.
chunks = split_words((root/"data"/"policies.txt").read_text(encoding="utf-8"), 35, 8, "policies")
vectors = embed_openai([item.content for item in chunks])
question = "¿Cómo recupero mi contraseña?"
# La consulta debe embebirse con el mismo modelo que se utilizó para el corpus.
results = top_k(embed_openai([question])[0], chunks, vectors, k=3)
# Persistimos pregunta y evidencia para que el resultado pueda revisarse posteriormente.
write_json(root/"data"/"expected"/"retrieval_results.json", {"question": question, "results": results})
for item in results:
    print(item["chunk"]["chunk_id"], round(item["score"], 3), item["chunk"]["source"])
