# Este archivo compara dos decisiones de chunking sobre exactamente el mismo corpus. Muestra
# que un chunk pequeno aporta precision y que overlap conserva continuidad entre fragmentos.
# Al ejecutarlo veras cantidad y contenido de los primeros chunks de ambas configuraciones.
# sys agrega la raiz para poder importar los datos compartidos.
import sys
# Path ubica esa raiz desde la ruta del script.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# RecursiveCharacterTextSplitter permite variar tamano y overlap de los chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# DOCUMENTS contiene el mismo corpus para comparar ambas configuraciones.
from shared.dataset import DOCUMENTS

# Chunks pequenos son mas especificos, pero pueden perder contexto si no hay overlap.
small_chunks = RecursiveCharacterTextSplitter(chunk_size=55, chunk_overlap=0).split_documents(DOCUMENTS)
print(f"Con chunk_size=55 y overlap=0 se crean {len(small_chunks)} chunks")
for chunk in small_chunks[:3]:
    print("-", chunk.page_content[:70])

# El overlap repite una parte del texto anterior para proteger ideas que cruzan el limite.
wide_chunks = RecursiveCharacterTextSplitter(chunk_size=90, chunk_overlap=20).split_documents(DOCUMENTS)
print(f"\nCon chunk_size=90 y overlap=20 se crean {len(wide_chunks)} chunks")
for chunk in wide_chunks[:3]:
    print("-", chunk.page_content[:70])

print("\nMas overlap preserva continuidad, pero tambien indexa y envia texto duplicado.")

# Resumen final: chunks pequenos encuentran detalles; chunks amplios conservan mas contexto.
# Overlap es un costo adicional que se justifica cuando las ideas cruzan limites de fragmentos.
