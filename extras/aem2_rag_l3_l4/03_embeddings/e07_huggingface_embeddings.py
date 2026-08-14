# Este archivo ensena a crear embeddings locales con Hugging Face y sentence-transformers.
# Necesita instalar langchain-huggingface y sentence-transformers; la primera ejecucion descarga
# el modelo elegido. Al terminar muestra el vector de una pregunta sin usar API key cloud.
# sys agrega la raiz del curso al path de imports.
import sys
# Path obtiene esa raiz desde la ubicacion actual.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# optional_import informa como instalar una dependencia que no esta disponible.
from shared.utils import optional_import

# HuggingFaceEmbeddings permite usar sentence-transformers con la misma interfaz
# embed_query que un proveedor cloud, pero el modelo se ejecuta en esta maquina.
if optional_import("langchain_huggingface", "langchain-huggingface sentence-transformers"):
    # HuggingFaceEmbeddings conecta modelos sentence-transformers con LangChain.
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector = embeddings.embed_query("Puedo trabajar desde casa?")
    print("Dimension:", len(vector), "primeras dimensiones:", vector[:5])
    print("El proveedor local elimina la API key, pero necesita modelo y memoria propios.")

# Resumen final: Hugging Face ofrece embeddings locales bajo el mismo contrato de LangChain.
# A cambio de autonomia y privacidad, el equipo debe descargar y ejecutar el modelo elegido.
