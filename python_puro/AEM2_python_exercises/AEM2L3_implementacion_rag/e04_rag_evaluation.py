"""E04: evaluar grounding y recuperación con golden cases."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import deterministic_embedding, read_json, split_words, top_k, write_json

root=Path(__file__).parent; chunks=split_words((root/"data"/"faq.txt").read_text(encoding="utf-8"),35,8,"faq")
vectors=[deterministic_embedding(c.content) for c in chunks]; report=[]
for case in read_json(root/"data"/"golden_cases.json"):
    results=top_k(deterministic_embedding(case["question"]),chunks,vectors,3)
    evidence=" ".join(x["chunk"]["content"].lower() for x in results)
    report.append({"question":case["question"],"grounded":all(k in evidence for k in case["keywords"]),"chunks":[x["chunk"]["chunk_id"] for x in results]})
write_json(root/"data"/"evaluation.json",report); print(report)

