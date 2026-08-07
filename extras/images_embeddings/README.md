# Ejercicio simple: embeddings de un PDF con texto e imágenes

Un embedding de texto normal no "ve" las imágenes de un PDF. La estrategia más simple
para no perderlas: describir cada imagen con un modelo vision (GPT-4o-mini) y agregar
esa descripción al texto de la página. Después se genera un único embedding de texto
por página, como en `extras/embeddings/manual`. No hace falta un modelo multimodal ni
cambiar de base de datos: el vector sigue siendo un embedding de texto normal.

## Instalar

```bash
& ".\.venv\Scripts\python.exe" -m pip install -r ".\images_embeddings\requirements.txt"
```

La key se toma del `.env` que ya existe en la raíz del proyecto.

## Antes de correr

Poné un PDF con texto e imágenes en `data/documento.pdf` (o cambiá `PDF_PATH` en el script).

## Ejecutar

```bash
& ".\.venv\Scripts\python.exe" ".\images_embeddings\ejemplo_pdf_imagenes.py"
```

Por cada página, el script imprime cuánto texto e imágenes tiene, la descripción que
generó para cada imagen, y al final la dimensión y un resumen del embedding de cada página.
