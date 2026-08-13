"""RAG local mínimo: loader → splitter → embeddings → vectorstore → retrieval.
Primero se inspecciona el contexto recuperado; un LLM solo se agrega después.
"""
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS

curso = Path(__file__).resolve().parents[1]
documentos = TextLoader(str(curso / "data" / "faq_empresa_saas.txt"), encoding="utf-8").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50).split_documents(documentos)
store = FAISS.from_documents(chunks, FakeEmbeddings(size=32))
resultados = store.similarity_search("¿Cómo contacto soporte?", k=3)

print("Chunks creados:", len(chunks))
for documento in resultados:
    print("-----")
    print(documento.page_content)
