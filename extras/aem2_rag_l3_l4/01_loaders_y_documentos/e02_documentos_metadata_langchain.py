# Este archivo presenta la unidad basica de LangChain: Document. Al ejecutarlo se ven los textos
# del corpus y la metadata que los acompana, para entender que contenido se busca y que datos
# se usan luego para filtrar o citar. No requiere modelos, red ni credenciales.
# sys permite encontrar los modulos compartidos al ejecutar este archivo directamente.
import sys
# Path obtiene la carpeta raiz a partir de la ubicacion del script.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS es el corpus de Documents LangChain usado en todos los ejercicios.
from shared.dataset import DOCUMENTS
# show_documents imprime contenido y metadata para observar la estructura.
from shared.utils import show_documents

# Un Document es la pieza que viaja por LangChain. page_content es evidencia
# recuperable; metadata aporta ID, categoria y fuente para filtros o citas.
show_documents(DOCUMENTS)

print("\nSeparar contenido y metadata permite actualizar los datos tecnicos sin tocar la evidencia.")

# Resumen final: Document es el contrato comun de LangChain para transportar evidencia.
# page_content responde preguntas y metadata permite filtrar, identificar y citar fuentes.
