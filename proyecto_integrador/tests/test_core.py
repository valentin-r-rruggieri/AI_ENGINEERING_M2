import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import clean_text, split_into_chunks
from src.models import RAGAnswer, RetrievedChunk


class CoreTests(unittest.TestCase):
    def test_clean_text_rejects_empty(self):
        with self.assertRaises(ValueError):
            clean_text("   \n ")

    def test_chunking_requires_twenty_chunks(self):
        text = "palabra " * 3000
        chunks = split_into_chunks(text, "test.txt", 100, 20)
        self.assertGreaterEqual(len(chunks), 20)
        self.assertTrue(all(chunk.content for chunk in chunks))

    def test_public_response_has_three_keys(self):
        chunk = RetrievedChunk(
            chunk_id="faq-001", content="Contenido", source="faq.txt",
            start_word=0, end_word=1, score=0.9,
        )
        answer = RAGAnswer(
            user_question="Pregunta", system_answer="Respuesta",
            chunks_related=[chunk],
        )
        self.assertEqual(
            set(answer.model_dump()),
            {"user_question", "system_answer", "chunks_related"},
        )


if __name__ == "__main__":
    unittest.main()

