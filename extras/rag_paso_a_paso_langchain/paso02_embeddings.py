"""Paso 2: ver que vector genera OpenAIEmbeddings para cada Document.

A diferencia del paso a paso manual, LangChain no obliga a guardar los
vectores en un archivo intermedio: el vector store (pasos 3 y 5) se encarga
de generarlos por dentro cuando lo armamos. Este paso es solo para ver la
dimension antes de que quede "escondida" dentro del vector store.
Docs: https://python.langchain.com/docs/integrations/text_embedding/openai/
"""
from comun import cargar_documentos, embeddings

documentos = cargar_documentos("01_documentos.json")

textos = [documento.page_content for documento in documentos]
vectores = embeddings.embed_documents(textos)

print(f"Embeddings generados: {len(vectores)} (dimension {len(vectores[0])})")
