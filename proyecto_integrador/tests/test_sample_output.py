import json
import unittest
from pathlib import Path


class SampleOutputTests(unittest.TestCase):
    def test_sample_outputs_respect_public_contract(self):
        path = Path(__file__).resolve().parents[1] / "outputs" / "sample_queries.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload), 3)
        expected = {"user_question", "system_answer", "chunks_related"}
        self.assertTrue(all(set(item) == expected for item in payload))
        self.assertTrue(all(2 <= len(item["chunks_related"]) <= 5 for item in payload))


if __name__ == "__main__":
    unittest.main()
