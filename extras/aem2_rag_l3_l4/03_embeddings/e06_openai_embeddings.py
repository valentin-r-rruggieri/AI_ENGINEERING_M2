# Este archivo usa OpenAIEmbeddings para convertir una pregunta en un vector semantico remoto.
# Necesita OPENAI_API_KEY en .env y, si falta, explica como configurarla sin exponer secretos.
# Al ejecutarlo con la clave correcta se observa la dimension y una muestra del embedding.
# sys habilita los imports compartidos desde una ejecucion directa.
import sys
# Path permite calcular la raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# needs_openai revisa OPENAI_API_KEY y evita una llamada cloud sin credenciales.
from shared.utils import needs_openai

# OpenAIEmbeddings implementa el contrato de LangChain: texto de entrada, vector de salida.
# La clave se lee desde .env; nunca se escribe dentro de un ejercicio.
if needs_openai():
    # OpenAIEmbeddings convierte texto en vectores mediante el proveedor OpenAI.
    from langchain_openai import OpenAIEmbeddings

    vector = OpenAIEmbeddings().embed_query("Cuantos dias de vacaciones tengo?")
    print("Dimension:", len(vector))
    print("Primeras dimensiones:", vector[:5])
    print("Cada dimension aporta una senal para comparar significado entre textos.")

# Resumen final: los embeddings cloud permiten comparar significado con una API de LangChain.
# La clave va en .env y el mismo proveedor debe vectorizar tanto corpus como preguntas.
