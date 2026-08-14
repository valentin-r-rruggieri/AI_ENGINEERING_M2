# Este archivo aplica RecursiveCharacterTextSplitter al corpus comun. Sirve para observar como
# LangChain divide Documents grandes en partes que se pueden recuperar con mas precision. Al
# ejecutarlo se imprimen los chunks y su indice; puedes cambiar tamano y overlap para comparar.
# sys habilita importar shared desde un script ejecutado de forma aislada.
import sys
# Path calcula la raiz del curso desde este archivo.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# RecursiveCharacterTextSplitter implementa el chunking recursivo de LangChain.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# DOCUMENTS aporta el corpus comun del curso.
from shared.dataset import DOCUMENTS
# show_documents deja visible el resultado del splitter.
from shared.utils import show_documents

# RecursiveCharacterTextSplitter intenta primero respetar parrafos y espacios.
# Asi genera fragmentos legibles antes de caer en cortes mas pequenos.
chunks = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=15).split_documents(DOCUMENTS)

# Agregar el indice deja visible el orden de los fragmentos de un mismo documento.
for index, chunk in enumerate(chunks):
    chunk.metadata["chunk_index"] = index

show_documents(chunks)
print("\nEl splitter es parte de la calidad de retrieval: cambia que evidencia puede encontrarse.")

# Resumen final: dividir bien un documento mejora la precision sin perder demasiada continuidad.
# El tamano de chunk y el overlap deben elegirse mirando consultas reales del dominio.
