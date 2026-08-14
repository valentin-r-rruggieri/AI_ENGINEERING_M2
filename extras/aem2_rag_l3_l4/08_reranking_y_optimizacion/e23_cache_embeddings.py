# Este archivo demuestra cachear embeddings, una optimizacion que evita vectorizar el mismo texto
# repetidas veces. Usa un proveedor local didactico y muestra HIT o MISS para cada consulta. Al
# ejecutarlo se ve que el segundo texto identico no vuelve a llamar al proveedor.
# sys permite importar el helper local sin instalar un paquete.
import sys
# Path encuentra la raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# KeywordEmbeddings simula un proveedor de vectores para observar el cache sin red.
from shared.utils import KeywordEmbeddings

# El cache guarda vectores por texto y evita repetir exactamente la misma llamada de embeddings.
embeddings, cache, calls = KeywordEmbeddings(), {}, 0
for text in ["Puedo trabajar remoto?", "Puedo trabajar remoto?", "Cuantos dias de vacaciones?"]:
    if text not in cache:
        cache[text] = embeddings.embed_query(text)
        calls += 1
        print("MISS:", text)
    else:
        print("HIT: ", text)

print("Llamadas reales al proveedor:", calls)
print("El cache debe invalidarse cuando cambia el modelo o la version del documento.")

# Resumen final: cachear embeddings evita trabajo repetido cuando el mismo texto aparece otra vez.
# La clave del cache debe incluir la version del modelo para no mezclar vectores incompatibles.
