# Este archivo ensena batching para embeddings: varios textos entran juntos en embed_documents.
# El ejemplo divide el corpus en lotes pequenos y muestra cuantos vectores entrega cada uno. Al
# ejecutarlo se entiende donde ajustar throughput, memoria y limites de un proveedor real.
# sys agrega los modulos shared al path de Python.
import sys
# Path calcula la raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS aporta los textos que se enviaran por lotes.
from shared.dataset import DOCUMENTS
# KeywordEmbeddings implementa embed_documents para mostrar el contrato de batch.
from shared.utils import KeywordEmbeddings

# embed_documents recibe una lista: agrupar textos reduce overhead frente a enviar uno por uno.
embeddings = KeywordEmbeddings()
batch_size = 2
for start in range(0, len(DOCUMENTS), batch_size):
    batch = DOCUMENTS[start : start + batch_size]
    vectors = embeddings.embed_documents([doc.page_content for doc in batch])
    print(f"Batch {start // batch_size + 1}: {len(batch)} documentos -> {len(vectors)} vectores")

print("El tamano de batch se ajusta segun memoria, limite de API y throughput buscado.")

# Resumen final: batching procesa varios documentos en una llamada y mejora el throughput.
# El lote no debe ser tan grande que exceda memoria local o limites del proveedor de embeddings.
