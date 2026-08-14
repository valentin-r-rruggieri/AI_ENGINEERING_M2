# Este archivo demuestra por que un indice no debe construirse desde cero en cada arranque.
# Crea una coleccion Chroma, la guarda dentro de storage y la abre otra vez para consultar.
# Al ejecutarlo veras la ruta persistida y una busqueda recuperada desde la nueva instancia.
# shutil borra el indice temporal anterior antes de recrear el ejemplo.
import shutil
# sys agrega la raiz del curso al buscador de modulos.
import sys
# Path permite construir rutas de almacenamiento.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Chroma crea y vuelve a abrir una coleccion vectorial persistente.
from langchain_chroma import Chroma

# DOCUMENTS es el contenido que se persiste en el indice.
from shared.dataset import DOCUMENTS
# KeywordEmbeddings vectoriza localmente, ROOT da la ruta y show_documents imprime la busqueda.
from shared.utils import KeywordEmbeddings, ROOT, show_documents

# Un directorio persistente evita calcular embeddings en cada ejecucion del programa.
directory = ROOT / "storage" / "chroma_e11"
if directory.exists():
    shutil.rmtree(directory)

# Se construye la coleccion una vez y luego se abre una nueva instancia desde disco.
Chroma.from_documents(DOCUMENTS, KeywordEmbeddings(), collection_name="policies", persist_directory=str(directory))
reloaded = Chroma(collection_name="policies", embedding_function=KeywordEmbeddings(), persist_directory=str(directory))

print("Indice guardado en:", directory)
show_documents(reloaded.similarity_search("horario de trabajo", k=1))
print("Al persistir, tambien hay que versionar el corpus y el modelo de embeddings.")

# Resumen final: persistir un indice evita recalcular embeddings cada vez que inicia el RAG.
# Un indice guardado solo es confiable si se conoce el corpus y modelo con que fue creado.
