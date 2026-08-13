"""CharacterTextSplitter: divide usando un separador literal.
Conviene cuando el texto ya tiene párrafos confiables.
# GUÍA DOCENTE
# CUÁNDO USAR: texto con un separador fiable, como párrafos o registros.
# DIFERENCIA: corta solo donde se indica; RecursiveCharacterTextSplitter prueba
# varios separadores y es más seguro para prosa irregular.
# EN CLASE: cambiar separator y observar cuándo un chunk supera el tamaño objetivo.
"""
from langchain_text_splitters import CharacterTextSplitter

texto = "Producto y planes.\n\nFacturación mensual.\n\nSoporte por email."
chunks = CharacterTextSplitter(separator="\n\n", chunk_size=30, chunk_overlap=5).split_text(texto)

print("Chunks:", len(chunks))
for chunk in chunks:
    print(repr(chunk))
