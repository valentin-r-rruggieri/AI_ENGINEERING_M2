"""RAG hecho a mano: chunking + embeddings + busqueda por coseno + respuesta."""
from pathlib import Path
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))

from documentos import DOCUMENTOS, PREGUNTA  # noqa: E402

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
client = OpenAI()

# 1. Chunking: partimos cada documento en lineas no vacias.
chunks = []
for doc in DOCUMENTOS:
    for linea in doc["text"].splitlines():
        if linea.strip():
            chunks.append({"source": doc["source"], "text": linea.strip()})
print(f"Chunks generados: {len(chunks)}")

# 2. Embeddings: le pedimos a OpenAI el vector de cada chunk.
# Docs: https://platform.openai.com/docs/guides/embeddings
# API ref: https://platform.openai.com/docs/api-reference/embeddings
textos = [chunk["text"] for chunk in chunks]
embeddings_response = client.embeddings.create(model=EMBEDDING_MODEL, input=textos)
embeddings = [np.array(item.embedding) for item in embeddings_response.data]
print(f"Cada embedding tiene {len(embeddings[0])} dimensiones")

# 3. Convertimos la pregunta del usuario en un embedding, igual que los chunks.
pregunta_embedding = np.array(
    client.embeddings.create(model=EMBEDDING_MODEL, input=PREGUNTA).data[0].embedding
)

# 4. Similitud coseno: comparamos la pregunta contra cada chunk y ordenamos.
# Docs de las funciones de NumPy usadas: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
# y https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
def coseno(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = [coseno(pregunta_embedding, emb) for emb in embeddings]
ranking = sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)
top_3 = ranking[:3]

print("\nTop 3 chunks mas parecidos a la pregunta:")
for chunk, score in top_3:
    print(f"  [{score:.4f}] ({chunk['source']}) {chunk['text']}")

# 5. RAG: mandamos la pregunta + los chunks recuperados al modelo de chat.
# Docs: https://platform.openai.com/docs/guides/text-generation
# API ref: https://platform.openai.com/docs/api-reference/chat
contexto = "\n".join(f"({chunk['source']}) {chunk['text']}" for chunk, _ in top_3)
respuesta = client.chat.completions.create(
    model=CHAT_MODEL,
    messages=[
        {"role": "system", "content": "Responde solo usando el contexto."},
        {"role": "user", "content": f"Pregunta: {PREGUNTA}\n\nContexto:\n{contexto}"},
    ],
)

print(f"\nPregunta: {PREGUNTA}")
print(f"Respuesta: {respuesta.choices[0].message.content}")
