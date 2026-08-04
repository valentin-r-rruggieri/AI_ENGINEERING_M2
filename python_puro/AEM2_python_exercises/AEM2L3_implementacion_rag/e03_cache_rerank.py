"""E03: caché de embeddings y reranking explicable."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import deterministic_embedding, split_words, top_k

cache={}
def embed_cached(text):
    key=" ".join(text.lower().split())
    return cache.setdefault(key,deterministic_embedding(key))

root=Path(__file__).parent
chunks=split_words((root/"data"/"faq.txt").read_text(encoding="utf-8"),35,8,"faq")
initial=top_k(embed_cached("¿Cómo recupero mi contraseña?"),chunks,[embed_cached(c.content) for c in chunks],3)
reranked=sorted(initial,key=lambda item:("contraseña" in item["chunk"]["content"].lower(),item["score"]),reverse=True)
print("cache entries:",len(cache)); print([x["chunk"]["chunk_id"] for x in reranked])

