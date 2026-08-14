# Este archivo combina busqueda vectorial con un filtro de metadata. Primero indexa todo el
# corpus, pero recupera solamente documentos de RRHH para una consulta sobre horario. Al
# ejecutarlo se ve como los filtros acotan contexto, permisos o dominios de conocimiento.
# sys permite importar el corpus y helpers desde un script independiente.
import sys
# Path resuelve la carpeta raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Chroma acepta filtros de metadata junto a la busqueda vectorial.
from langchain_chroma import Chroma

# DOCUMENTS incluye la categoria con la que se filtrara la busqueda.
from shared.dataset import DOCUMENTS
# KeywordEmbeddings genera vectores locales y show_documents imprime el resultado filtrado.
from shared.utils import KeywordEmbeddings, show_documents

# La metadata permite acotar el espacio de busqueda antes de construir el contexto.
store = Chroma.from_documents(DOCUMENTS, KeywordEmbeddings(), collection_name="e15")
results = store.similarity_search("Cual es el horario?", k=3, filter={"category": "rrhh"})
show_documents(results)
print("Los filtros son importantes para permisos, tenants y dominios de conocimiento separados.")

# Resumen final: la metadata no solo sirve para citar; tambien limita que conocimiento se consulta.
# Filtrar temprano reduce ruido y es una base importante para control de acceso en RAG.
