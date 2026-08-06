"""E02: fragmentar y preservar procedencia."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import split_words, write_json

root = Path(__file__).parent
text = (root / "data" / "policies.txt").read_text(encoding="utf-8")
# Cada chunk conserva su fuente y el rango de palabras para poder auditarlo después.
chunks = split_words(text, chunk_size=35, overlap=8, source="policies")
# El archivo generado permite inspeccionar el resultado sin volver a fragmentar.
write_json(root / "data" / "expected" / "chunks.json", [chunk.to_dict() for chunk in chunks])
for chunk in chunks:
    # La vista previa permite comprobar visualmente el overlap entre fragmentos.
    print(chunk.chunk_id, chunk.start_word, chunk.end_word, chunk.content[:55])
