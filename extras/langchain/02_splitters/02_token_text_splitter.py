# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""TokenTextSplitter: limita por tokens, no por caracteres.
Se usa cuando el presupuesto de contexto o costo depende de tokens.
# GUÍA DOCENTE
# CUÁNDO USAR: cuando se controla contexto, costo o límite exacto del modelo.
# DIFERENCIA: caracteres visibles no equivalen a tokens; este splitter mide lo
# mismo que el modelo, aunque puede cortar una oración menos elegantemente.
# EN CLASE: comparar un mismo texto con splitter por caracteres y por tokens.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_text_splitters import TokenTextSplitter

texto = "LangChain organiza documentos, chunks, embeddings y retrieval. " * 12
chunks = TokenTextSplitter(chunk_size=30, chunk_overlap=5).split_text(texto)

print("Chunks:", len(chunks))
for chunk in chunks[:3]:
    print(chunk)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
