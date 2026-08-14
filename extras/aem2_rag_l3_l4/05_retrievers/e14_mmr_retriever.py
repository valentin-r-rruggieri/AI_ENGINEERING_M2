# Este archivo usa MMR para recuperar documentos relevantes pero poco redundantes. Es util cuando
# una pregunta abarca varias ideas y el contexto no debe llenarse con casi el mismo texto. Al
# ejecutarlo se observa la evidencia elegida y se puede cambiar lambda_mult para experimentar.
# sys agrega shared al path de imports.
import sys
# Path ayuda a localizar la raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Chroma ofrece el tipo de busqueda MMR a traves de su retriever.
from langchain_chroma import Chroma

# DOCUMENTS contiene los candidatos que el retriever evaluara.
from shared.dataset import DOCUMENTS
# KeywordEmbeddings vectoriza localmente y show_documents deja ver diversidad y metadata.
from shared.utils import KeywordEmbeddings, show_documents

# MMR combina relevancia y diversidad: evita llenar el contexto con chunks casi iguales.
retriever = Chroma.from_documents(DOCUMENTS, KeywordEmbeddings(), collection_name="e14").as_retriever(
    search_type="mmr", search_kwargs={"k": 2, "fetch_k": 5, "lambda_mult": 0.5}
)
show_documents(retriever.invoke("politicas de empleados vacaciones trabajo horario"))
print("Cambiar lambda_mult permite priorizar similitud o variedad entre los resultados.")

# Resumen final: MMR recupera evidencia relevante sin repetir demasiadas veces la misma idea.
# Es especialmente util cuando el contexto debe cubrir varios aspectos de una consulta amplia.
