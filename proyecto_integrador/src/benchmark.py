from __future__ import annotations

import json
import time

from .config import Settings
from .embeddings import embed_texts
from .stores import create_store


def benchmark() -> list[dict[str, object]]:
    settings = Settings.from_env()
    queries_path = settings.data_dir / "golden_cases.json"
    cases = json.loads(queries_path.read_text(encoding="utf-8"))
    report: list[dict[str, object]] = []
    for case in cases:
        vector = embed_texts([case["question"]], settings)[0]
        for backend in ("chroma", "faiss"):
            started = time.perf_counter()
            results = create_store(backend, settings.storage_dir).search(vector, settings.top_k)
            elapsed_ms = (time.perf_counter() - started) * 1_000
            retrieved_ids = [item.chunk_id for item in results]
            report.append({
                "backend": backend,
                "question": case["question"],
                "latency_ms": round(elapsed_ms, 3),
                "retrieved_ids": retrieved_ids,
                "expected_keywords": case["expected_keywords"],
                "keyword_hits": sum(
                    any(keyword.lower() in item.content.lower() for item in results)
                    for keyword in case["expected_keywords"]
                ),
            })
    output = settings.storage_dir / "benchmark_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    for row in benchmark():
        print(row)

