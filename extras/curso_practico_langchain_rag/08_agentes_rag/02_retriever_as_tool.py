"""create_retriever_tool hace que un agente pueda pedir evidencia documental.
La tool devuelve contexto; el agente decide si debe llamarla.
"""
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

store = FAISS.from_texts(["La licencia Pro admite 10 usuarios.", "El soporte atiende por correo."], FakeEmbeddings(size=16))
herramienta = create_retriever_tool(store.as_retriever(search_kwargs={"k": 1}), "buscar_faq", "Busca evidencia en la FAQ.")

print(herramienta.invoke({"query": "¿Cuántos usuarios permite Pro?"}))
