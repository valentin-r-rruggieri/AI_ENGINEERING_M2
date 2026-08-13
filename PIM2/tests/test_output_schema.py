from __future__ import annotations

import json
from pathlib import Path


def test_sample_queries_match_the_public_contract() -> None:
    output_path = Path(__file__).resolve().parents[1] / "outputs" / "sample_queries.json"
    samples = json.loads(output_path.read_text(encoding="utf-8"))
    expected_keys = {"user_question", "system_answer", "chunks_related"}

    assert len(samples) == 3
    assert all(set(sample) == expected_keys for sample in samples)
    assert all(2 <= len(sample["chunks_related"]) <= 5 for sample in samples)
    assert all(sample["system_answer"].strip() for sample in samples)
    assert all(
        set(chunk) == {"chunk_id", "content"}
        for sample in samples
        for chunk in sample["chunks_related"]
    )
