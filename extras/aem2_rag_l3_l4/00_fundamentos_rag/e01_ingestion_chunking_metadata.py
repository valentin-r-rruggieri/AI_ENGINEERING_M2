# Este archivo muestra el primer paso real de un RAG: transformar un archivo de politicas
# en Documents y chunks de LangChain. Al ejecutarlo veras cada fragmento junto con la metadata
# que permite conocer su fuente, version y posicion. No requiere API key ni modelo descargado.
# sys permite agregar la raiz del curso al buscador de modulos de Python.
import sys
# Path construye rutas portables hacia data y shared.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# TextLoader carga un archivo de texto y lo transforma en Documents de LangChain.
from langchain_community.document_loaders import TextLoader
# RecursiveCharacterTextSplitter corta texto respetando separadores naturales.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ROOT apunta a la carpeta del curso y show_documents imprime cada Document con su metadata.
from shared.utils import ROOT, show_documents

# TextLoader convierte el TXT en un Document de LangChain. El texto queda en
# page_content y LangChain agrega la ruta de origen dentro de metadata.
document = TextLoader(str(ROOT / "data" / "policies.txt"), encoding="utf-8").load()[0]

# Un chunk de 120 caracteres conserva contexto; 30 caracteres de overlap evitan
# que una idea se corte por completo entre dos fragmentos consecutivos.
splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=30)
chunks = splitter.split_documents([document])

# Esta metadata es la que despues permite filtrar, citar y saber que version se indexo.
for index, chunk in enumerate(chunks):
    chunk.metadata.update({"chunk_index": index, "source": "policies.txt", "document_version": "v1"})

show_documents(chunks)
print("\nCada chunk conserva el texto y la procedencia necesaria para una respuesta RAG trazable.")

# Resumen final: antes de indexar un corpus hay que fragmentarlo y enriquecerlo con metadata.
# Esa trazabilidad permite saber que texto sostendra cada respuesta posterior del RAG.
