# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""PythonCodeTextSplitter: divide código priorizando funciones y clases.
Evita separar una firma de función de su cuerpo cuando el tamaño lo permite.
# GUÍA DOCENTE
# CUÁNDO USAR: repositorios o documentación técnica con código Python.
# DIFERENCIA: prioriza imports, clases y funciones; un splitter de prosa puede
# separar una función de su cuerpo y empeorar la recuperación.
# EN CLASE: aumentar chunk_size y verificar qué bloques permanecen completos.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_text_splitters import PythonCodeTextSplitter

codigo = "import os\n\ndef saludar(nombre):\n    return f'Hola {nombre}'\n\nclass Servicio:\n    def ejecutar(self):\n        return saludar('equipo')"
chunks = PythonCodeTextSplitter(chunk_size=55, chunk_overlap=0).split_text(codigo)

for chunk in chunks:
    print("-----")
    print(chunk)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
