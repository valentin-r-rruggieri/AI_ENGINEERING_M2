"""E04: recuperar candidatos y aplicar filtro de metadata."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import deterministic_embedding, split_words, top_k

root=Path(__file__).parent
chunks=split_words((root/"data"/"corpus.txt").read_text(encoding="utf-8"),20,4,"corpus")
# En un caso real esta metadata viene del documento o del sistema de permisos.
for chunk in chunks:
    content=chunk.content.lower()
    chunk.metadata["department"]="rrhh" if "vacaciones" in content or "licencias" in content else "tecnologia"
vectors=[deterministic_embedding(c.content) for c in chunks]
# Primero recuperamos por significado; después aplicamos la restricción estructurada.
candidates=top_k(deterministic_embedding("vacaciones"),chunks,vectors,min(5,len(chunks)))
filtered=[item for item in candidates if item["chunk"]["metadata"]["department"]=="rrhh"]
print([item["chunk"]["chunk_id"] for item in filtered])
