# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""DirectoryLoader: carga todos los archivos de una carpeta.
Elegir glob evita indexar formatos que no corresponden al loader elegido.
# GUÍA DOCENTE
# CUÁNDO USAR: cuando el corpus tiene muchos archivos del mismo formato.
# DIFERENCIA: glob selecciona qué cargar; loader_cls define cómo interpretar cada
# archivo. No usarlo sin glob en una carpeta con formatos mezclados.
# EN CLASE: cambiar **/*.txt y discutir qué archivos entrarían al índice.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.document_loaders import DirectoryLoader, TextLoader

curso = Path(__file__).resolve().parents[1]
documentos = DirectoryLoader(
    str(curso / "data"),
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
).load()

print("Archivos cargados:", len(documentos))
for documento in documentos:
    print(documento.metadata["source"], "|", len(documento.page_content), "caracteres")

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
