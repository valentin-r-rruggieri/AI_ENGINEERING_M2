# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""PyPDFLoader: carga un PDF digital página por página.
metadata['page'] permite citar la página. Un PDF escaneado necesita OCR externo.
# GUÍA DOCENTE
# CUÁNDO USAR: PDF digital cuyo texto se puede seleccionar.
# DIFERENCIA: PyPDFLoader genera un Document por página; un PDF escaneado necesita
# OCR antes, porque no contiene texto real que extraer.
# EN CLASE: citar metadata['page'] y comparar una página con un chunk posterior.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.document_loaders import PyPDFLoader

archivo = Path(__file__).resolve().parents[3] / "RAG_Chunking_Engineering.pdf"

if archivo.exists():
    documentos = PyPDFLoader(str(archivo)).load()
    print("Páginas:", len(documentos))
    print("Página:", documentos[0].metadata["page"] + 1)
    print(documentos[0].page_content[:300])
else:
    print("Coloca un PDF junto al curso para ejecutar este ejemplo.")

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
