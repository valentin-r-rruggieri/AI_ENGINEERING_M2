# Este archivo muestra donde conectar OpenAIEmbedding a LlamaIndex. Solo se ejecuta si la
# integracion y OPENAI_API_KEY estan disponibles, para no exponer claves ni hacer llamadas por
# accidente. Con la configuracion correcta imprime la dimension de un embedding cloud.

# os lee la clave desde variables de entorno.
import os
# importlib.util detecta la integracion opcional.
import importlib.util

if importlib.util.find_spec("llama_index.embeddings.openai") and os.getenv("OPENAI_API_KEY"):
    # OpenAIEmbedding vectoriza texto mediante el proveedor OpenAI.
    from llama_index.embeddings.openai import OpenAIEmbedding
    vector = OpenAIEmbedding().get_query_embedding("Cuantos dias de vacaciones tengo?")
    print("Dimension:", len(vector))
else:
    print("Configura OPENAI_API_KEY e instala llama-index-embeddings-openai para este ejemplo.")

# Resumen final: embeddings cloud se conectan a LlamaIndex sin cambiar el resto del pipeline.
