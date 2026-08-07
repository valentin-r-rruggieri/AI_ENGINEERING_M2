"""Paso 6: filtrar los resultados de FAISS por metadata (por ejemplo, por departamento).

La similitud semantica no sabe de permisos: primero buscamos por parecido,
despues aplicamos la regla de negocio (que departamento puede ver que chunk).
LangChain FAISS acepta un filtro de metadata directamente en la busqueda.
Docs: https://python.langchain.com/docs/integrations/vectorstores/faiss/
"""
from comun import PREGUNTA
from paso05_vectorstore_faiss import vectorstore


def buscar_con_filtro(pregunta: str, department: str, top_k: int = 3) -> list[tuple]:
    """Le pide directamente a FAISS que solo considere documentos de ese departamento."""
    return vectorstore.similarity_search_with_score(pregunta, k=top_k, filter={"department": department})


if __name__ == "__main__":
    for department in ["IT", "Finance", "Security"]:
        print(f"\nPregunta: {PREGUNTA}  (usuario de {department})")
        resultados = buscar_con_filtro(PREGUNTA, department)
        if not resultados:
            print("  (sin resultados para este departamento)")
        for documento, score in resultados:
            print(f"  [{score:.4f}] ({documento.metadata['source']}) {documento.page_content}")
