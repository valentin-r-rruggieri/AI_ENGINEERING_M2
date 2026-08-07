"""Paso 5: reemplazar el InMemoryVectorStore del paso 3 por un indice FAISS persistido.

Mismo resultado que el paso 3, pero FAISS puede escalar a millones de vectores
y se guarda en disco con save_local/load_local.
Docs: https://python.langchain.com/docs/integrations/vectorstores/faiss/
"""
from langchain_community.vectorstores import FAISS

from comun import ESTADO, PREGUNTA, cargar_documentos, embeddings

documentos = cargar_documentos("01_documentos.json")

INDICE_PATH = ESTADO / "05_faiss_index"
if INDICE_PATH.exists():
    vectorstore = FAISS.load_local(str(INDICE_PATH), embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(documentos, embeddings)
    vectorstore.save_local(str(INDICE_PATH))

print(f"Indice FAISS listo, guardado en {INDICE_PATH}")


def buscar_faiss(pregunta: str, top_k: int = 3) -> list[tuple]:
    """Busca los top_k documentos mas parecidos usando el indice FAISS."""
    return vectorstore.similarity_search_with_score(pregunta, k=top_k)


if __name__ == "__main__":
    print(f"\nPregunta: {PREGUNTA}")
    for documento, score in buscar_faiss(PREGUNTA):
        print(f"  [{score:.4f}] ({documento.metadata['source']}) {documento.page_content}")
