"""RecursiveJsonSplitter: divide JSON conservando sus claves.
Es preferible a convertir el JSON completo a texto antes de fragmentarlo.
# GUÍA DOCENTE
# CUÁNDO USAR: JSON anidado que debe conservar claves y contexto de objetos.
# DIFERENCIA: no corta una serialización plana arbitrariamente; mantiene rutas
# estructurales. Para una lista de registros simples, JSONLoader puede bastar.
# EN CLASE: agregar una clave anidada y observar cómo aparece en los chunks.
"""
from langchain_text_splitters import RecursiveJsonSplitter

datos = {"planes": [{"nombre": "Pro", "usuarios": 10}, {"nombre": "Team", "usuarios": 50}], "soporte": {"canal": "email"}}
documentos = RecursiveJsonSplitter(max_chunk_size=70).create_documents([datos])

for documento in documentos:
    print(documento.page_content)
