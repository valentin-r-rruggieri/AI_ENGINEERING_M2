"""MMR combina relevancia con diversidad.
fetch_k reúne candidatos; k conserva resultados; lambda_mult regula diversidad.
# GUÍA DOCENTE
# CUÁNDO USAR: cuando top-k devuelve chunks casi idénticos.
# DIFERENCIA: similarity maximiza cercanía; MMR combina cercanía y diversidad.
# fetch_k reúne candidatos, k devuelve resultados y lambda_mult ajusta balance.
# EN CLASE: comparar las dos listas para la misma pregunta.
"""
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS

store = FAISS.from_texts(["Plan Pro mensual.", "Plan Pro anual.", "Soporte por email.", "Seguridad de cuenta."], FakeEmbeddings(size=20))
retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4, "lambda_mult": 0.5})
documentos = retriever.invoke("plan")

for documento in documentos:
    print(documento.page_content)
