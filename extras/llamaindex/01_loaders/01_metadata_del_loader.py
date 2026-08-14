# Este archivo inspecciona la metadata que produce un loader. La procedencia es importante porque
# luego permite filtrar y citar la evidencia recuperada. Al ejecutarlo se muestran las claves que
# SimpleDirectoryReader conserva junto con el contenido del archivo cargado.

# Path construye la ruta absoluta del archivo de datos.
from pathlib import Path
# SimpleDirectoryReader carga el archivo y agrega metadata de origen.
from llama_index.core import SimpleDirectoryReader

file_path = Path(__file__).resolve().parents[1] / "data" / "faq_empresa.txt"
document = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()[0]
print("Metadata:", document.metadata)

# Resumen final: conservar metadata desde la carga hace posible explicar de donde salio una respuesta.
