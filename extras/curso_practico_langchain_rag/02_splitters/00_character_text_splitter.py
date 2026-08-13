"""CharacterTextSplitter: divide usando un separador literal.
Conviene cuando el texto ya tiene párrafos confiables.
"""
from langchain_text_splitters import CharacterTextSplitter

texto = "Producto y planes.\n\nFacturación mensual.\n\nSoporte por email."
chunks = CharacterTextSplitter(separator="\n\n", chunk_size=30, chunk_overlap=5).split_text(texto)

print("Chunks:", len(chunks))
for chunk in chunks:
    print(repr(chunk))
