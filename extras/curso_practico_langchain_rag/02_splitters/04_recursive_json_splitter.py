"""RecursiveJsonSplitter: divide JSON conservando sus claves.
Es preferible a convertir el JSON completo a texto antes de fragmentarlo.
"""
from langchain_text_splitters import RecursiveJsonSplitter

datos = {"planes": [{"nombre": "Pro", "usuarios": 10}, {"nombre": "Team", "usuarios": 50}], "soporte": {"canal": "email"}}
documentos = RecursiveJsonSplitter(max_chunk_size=70).create_documents([datos])

for documento in documentos:
    print(documento.page_content)
