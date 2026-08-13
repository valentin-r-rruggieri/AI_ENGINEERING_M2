"""PythonCodeTextSplitter: divide código priorizando funciones y clases.
Evita separar una firma de función de su cuerpo cuando el tamaño lo permite.
"""
from langchain_text_splitters import PythonCodeTextSplitter

codigo = "import os\n\ndef saludar(nombre):\n    return f'Hola {nombre}'\n\nclass Servicio:\n    def ejecutar(self):\n        return saludar('equipo')"
chunks = PythonCodeTextSplitter(chunk_size=55, chunk_overlap=0).split_text(codigo)

for chunk in chunks:
    print("-----")
    print(chunk)
