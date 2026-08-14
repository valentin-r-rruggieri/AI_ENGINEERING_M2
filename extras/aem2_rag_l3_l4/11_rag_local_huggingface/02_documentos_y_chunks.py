# Este archivo prepara la evidencia antes de usar cualquier modelo: toma Documents, los divide
# en chunks y conserva su metadata. No necesita Hugging Face ni FAISS, por eso sirve para probar
# la preparacion del corpus primero. Al ejecutarlo se ven los fragmentos que luego se indexaran.
# sys habilita imports compartidos al ejecutar este archivo de forma directa.
import sys
# Path resuelve la carpeta raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# RecursiveCharacterTextSplitter fragmenta Documents antes de generar embeddings.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# DOCUMENTS aporta el corpus comun, con contenido y metadata ya preparados.
from shared.dataset import DOCUMENTS
# show_documents imprime los chunks para inspeccionarlos antes de indexar.
from shared.utils import show_documents

# Los chunks son la unidad que se vectoriza y se recupera; no se vectoriza el archivo entero.
splitter = RecursiveCharacterTextSplitter(chunk_size=90, chunk_overlap=20)
chunks = splitter.split_documents(DOCUMENTS)

for index, chunk in enumerate(chunks):
    # El indice de chunk permite volver al fragmento exacto cuando se cite una respuesta.
    chunk.metadata["chunk_index"] = index

show_documents(chunks)
print("El tamano y overlap se ajustan con consultas reales, no solo mirando el texto.")

# Resumen final: el RAG local tambien depende de documentos bien fragmentados y trazables.
# Antes de elegir un modelo hay que confirmar que los chunks contienen unidades de evidencia utiles.
