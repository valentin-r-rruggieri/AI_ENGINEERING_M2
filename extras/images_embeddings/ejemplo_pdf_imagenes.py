"""RAG multimodal simple: un PDF con texto e imagenes, en un solo embedding.

Estrategia (la mas facil para empezar): a cada imagen del PDF le pedimos a un
modelo vision que la describa en una oracion, y esa descripcion se agrega junto
al texto de la pagina. Despues generamos UN embedding de texto por pagina
(texto + descripciones). No hace falta un modelo multimodal ni una base de
datos especial: FAISS/Chroma/pgvector reciben el mismo embedding de texto de
siempre.
"""
from pathlib import Path
import base64

import pymupdf  # docs: https://pymupdf.readthedocs.io/en/latest/
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

PDF_PATH = Path(__file__).resolve().parent / "data" / "documento.pdf"
VISION_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

client = OpenAI()

if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"No encontre {PDF_PATH}. Poné ahi un PDF con texto e imagenes "
        "(o cambia PDF_PATH por la ruta de tu archivo)."
    )

# 1. Abrimos el PDF y recorremos pagina por pagina.
documento = pymupdf.open(PDF_PATH)
print(f"PDF: {PDF_PATH.name} ({documento.page_count} paginas)")

chunks = []
for numero_pagina, pagina in enumerate(documento, start=1):
    texto = pagina.get_text().strip()
    imagenes = pagina.get_images(full=True)
    print(f"Pagina {numero_pagina}: {len(texto)} caracteres de texto, {len(imagenes)} imagenes")

    # 2. Cada imagen se describe con un modelo vision, asi el embedding "ve" lo que muestra.
    # Docs: https://platform.openai.com/docs/guides/vision
    descripciones = []
    for indice, imagen_info in enumerate(imagenes, start=1):
        xref = imagen_info[0]
        imagen_bytes = documento.extract_image(xref)["image"]
        imagen_b64 = base64.b64encode(imagen_bytes).decode("utf-8")

        respuesta = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describi en una oracion que muestra esta imagen."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{imagen_b64}"}},
                    ],
                }
            ],
        )
        descripcion = respuesta.choices[0].message.content.strip()
        descripciones.append(descripcion)
        print(f"  imagen {indice}: {descripcion}")

    # 3. El chunk de la pagina combina el texto original con las descripciones de sus imagenes.
    partes = [texto] + [f"Descripcion de imagen: {d}" for d in descripciones]
    texto_enriquecido = "\n".join(parte for parte in partes if parte)

    if texto_enriquecido:
        chunks.append({"page": numero_pagina, "text": texto_enriquecido})

documento.close()

# 4. Un embedding de texto por pagina, ya con la info de texto + imagenes combinada.
# Docs: https://platform.openai.com/docs/guides/embeddings
textos = [chunk["text"] for chunk in chunks]
embeddings_response = client.embeddings.create(model=EMBEDDING_MODEL, input=textos)
for chunk, item in zip(chunks, embeddings_response.data):
    chunk["embedding"] = item.embedding

print(f"\nEmbeddings generados: {len(chunks)} (dimension {len(chunks[0]['embedding'])})")
for chunk in chunks:
    resumen = chunk["text"][:80].replace("\n", " ")
    print(f"Pagina {chunk['page']}: {resumen}...")
