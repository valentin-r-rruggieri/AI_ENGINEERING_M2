"""Paso 4: por que normalizar antes de usar producto interno (la "trampa de la magnitud").

Armamos un Embeddings de juguete (la misma clase base que extiende OpenAIEmbeddings)
que devuelve vectores fijos, y usamos InMemoryVectorStore -igual que en el paso 3-
para ver, con herramientas puramente de LangChain, como el coseno (lo que usa
InMemoryVectorStore por dentro) puede dar un ganador distinto al del producto
interno sin normalizar.
Docs: https://python.langchain.com/docs/concepts/embedding_models/
"""
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


class EmbeddingsDeJuguete(Embeddings):
    """Devuelve vectores 2D fijos, para ver la trampa de la magnitud sin llamar a la API."""

    VECTORES = {
        "chunk largo": [10.0, 1.0],  # vector "grande", pero apunta poco hacia la pregunta
        "chunk corto": [2.0, 2.0],  # vector chico, pero apunta justo hacia la pregunta
        "pregunta": [1.0, 1.0],
    }

    def embed_documents(self, textos: List[str]) -> List[List[float]]:
        return [self.VECTORES[texto] for texto in textos]

    def embed_query(self, texto: str) -> List[float]:
        return self.VECTORES[texto]


vectorstore_juguete = InMemoryVectorStore.from_texts(
    ["chunk largo", "chunk corto"], embedding=EmbeddingsDeJuguete()
)

print("Con InMemoryVectorStore (usa similitud coseno por dentro):")
for documento, score in vectorstore_juguete.similarity_search_with_score("pregunta", k=2):
    print(f"  {documento.page_content}: coseno={score:.2f}")

print("\nCon producto interno sin normalizar (lo que haria un indice mal configurado):")
pregunta = np.array(EmbeddingsDeJuguete.VECTORES["pregunta"])
for nombre in ["chunk largo", "chunk corto"]:
    vector = np.array(EmbeddingsDeJuguete.VECTORES[nombre])
    print(f"  {nombre}: producto_interno={vector @ pregunta:.1f}")

print(
    "\n-> sin normalizar, 'chunk largo' gana (11.0 > 4.0) aunque InMemoryVectorStore, "
    "que si normaliza, diga que 'chunk corto' es mejor (1.00 vs 0.77)."
)
