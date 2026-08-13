"""TokenTextSplitter: limita por tokens, no por caracteres.
Se usa cuando el presupuesto de contexto o costo depende de tokens.
# GUÍA DOCENTE
# CUÁNDO USAR: cuando se controla contexto, costo o límite exacto del modelo.
# DIFERENCIA: caracteres visibles no equivalen a tokens; este splitter mide lo
# mismo que el modelo, aunque puede cortar una oración menos elegantemente.
# EN CLASE: comparar un mismo texto con splitter por caracteres y por tokens.
"""
from langchain_text_splitters import TokenTextSplitter

texto = "LangChain organiza documentos, chunks, embeddings y retrieval. " * 12
chunks = TokenTextSplitter(chunk_size=30, chunk_overlap=5).split_text(texto)

print("Chunks:", len(chunks))
for chunk in chunks[:3]:
    print(chunk)
