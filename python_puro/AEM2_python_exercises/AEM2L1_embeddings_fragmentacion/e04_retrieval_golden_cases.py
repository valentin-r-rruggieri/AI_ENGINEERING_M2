"""E04: evaluar Top-K sin consumir API, con embedding de test."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import deterministic_embedding, read_json, split_words, top_k

root = Path(__file__).parent
chunks = split_words((root/"data"/"policies.txt").read_text(encoding="utf-8"), 35, 8, "policies")
vectors = [deterministic_embedding(chunk.content) for chunk in chunks]
for case in read_json(root/"data"/"golden_cases.json"):
    found = top_k(deterministic_embedding(case["question"]), chunks, vectors, k=3)
    hit = any(keyword.lower() in item["chunk"]["content"].lower() for item in found for keyword in case["keywords"])
    print(case["question"], "OK" if hit else "REVISAR", [item["chunk"]["chunk_id"] for item in found])

