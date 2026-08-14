# Este archivo carga un TXT desde data usando el loader nativo SimpleDirectoryReader. Sirve para
# transformar archivos locales en Documents antes de chunking e indexacion. Al ejecutarlo muestra
# texto y metadata de los documentos cargados, sin necesidad de LLM ni API key.

# sys habilita imports del curso al ejecutar este archivo directamente.
import sys
# Path localiza la carpeta data del curso.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# SimpleDirectoryReader carga archivos locales como Documents de LlamaIndex.
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(input_files=[str(Path(__file__).resolve().parents[1] / "data" / "faq_empresa.txt")]).load_data()
print(documents[0].get_content()[:220])
print(documents[0].metadata)

# Resumen final: un loader convierte formatos externos en Documents que el resto del RAG entiende.
