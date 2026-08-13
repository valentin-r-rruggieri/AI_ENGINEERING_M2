"""BM25Retriever recupera por términos exactos, sin embeddings.
Es útil para códigos, nombres e identificadores; se combina con vectores en búsqueda híbrida.
"""
from langchain_core.documents import Document

try:
    from langchain_community.retrievers import BM25Retriever
    retriever = BM25Retriever.from_documents([Document(page_content="Error E401 de acceso."), Document(page_content="Soporte de facturación.")])
    print(retriever.invoke("E401")[0].page_content)
except ImportError:
    print("Instala BM25: pip install rank-bm25")
