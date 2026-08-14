# Este archivo crea el primer componente real del RAG local: embeddings con un modelo de Hugging
# Face. Necesita internet solo la primera vez para descargar el modelo y puede usar CPU. Al
# ejecutarlo se imprime la dimension y una muestra del vector de una pregunta en espanol.
# sys permite importar los helpers del curso desde este script independiente.
import sys
# Path encuentra la raiz del curso a partir de esta carpeta.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# optional_import informa la instalacion necesaria sin romper el recorrido.
from shared.utils import optional_import

# HuggingFaceEmbeddings expone un modelo sentence-transformers con la interfaz Embeddings de LangChain.
if optional_import("langchain_huggingface", "langchain-huggingface sentence-transformers"):
    # HuggingFaceEmbeddings carga y usa un modelo de embeddings que corre localmente.
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # embed_query recibe una pregunta y devuelve su vector numerico.
    vector = embeddings.embed_query("Puedo trabajar desde casa?")
    print("Modelo:", model_name)
    print("Dimension:", len(vector))
    print("Primeras dimensiones:", vector[:5])
    print("El mismo objeto tambien ofrece embed_documents para indexar varios textos.")

# Resumen final: HuggingFaceEmbeddings produce vectores locales que se usan en retrieval semantico.
# El mismo modelo debe utilizarse para embeddings de documentos y para embeddings de preguntas.
