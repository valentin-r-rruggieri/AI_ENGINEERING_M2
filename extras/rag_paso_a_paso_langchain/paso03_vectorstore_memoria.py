"""Paso 3: indexar los documentos en un InMemoryVectorStore (equivalente al k-NN exacto).

InMemoryVectorStore compara la pregunta contra todos los vectores en memoria
usando similitud coseno: es la version LangChain de la busqueda lineal del
paso a paso manual.
Docs: https://python.langchain.com/docs/integrations/vectorstores/in_memory/
"""
from langchain_core.vectorstores import InMemoryVectorStore

from comun import PREGUNTA, cargar_documentos, embeddings

documentos = cargar_documentos("01_documentos.json")
vectorstore = InMemoryVectorStore.from_documents(documentos, embeddings)


def buscar_exacto(pregunta: str, top_k: int = 3) -> list[tuple]:
    """Compara la pregunta contra TODOS los documentos y devuelve los top_k mas parecidos."""
    return vectorstore.similarity_search_with_score(pregunta, k=top_k)


if __name__ == "__main__":
    print(f"Pregunta: {PREGUNTA}")
    for documento, score in buscar_exacto(PREGUNTA):
        print(f"  [{score:.4f}] ({documento.metadata['source']}) {documento.page_content}")
