# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""RecursiveJsonSplitter: divide JSON conservando sus claves.
Es preferible a convertir el JSON completo a texto antes de fragmentarlo.
# GUÍA DOCENTE
# CUÁNDO USAR: JSON anidado que debe conservar claves y contexto de objetos.
# DIFERENCIA: no corta una serialización plana arbitrariamente; mantiene rutas
# estructurales. Para una lista de registros simples, JSONLoader puede bastar.
# EN CLASE: agregar una clave anidada y observar cómo aparece en los chunks.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_text_splitters import RecursiveJsonSplitter

datos = {"planes": [{"nombre": "Pro", "usuarios": 10}, {"nombre": "Team", "usuarios": 50}], "soporte": {"canal": "email"}}
documentos = RecursiveJsonSplitter(max_chunk_size=70).create_documents([datos])

for documento in documentos:
    print(documento.page_content)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
