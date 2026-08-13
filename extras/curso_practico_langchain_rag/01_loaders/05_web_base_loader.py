"""WebBaseLoader: carga texto de una página HTML pública.
Úsalo para una URL puntual y revisa permisos, limpieza y contenido duplicado.
# GUÍA DOCENTE
# CUÁNDO USAR: una página pública y estable que se puede descargar legalmente.
# DIFERENCIA: recupera HTML convertido a texto; no sustituye un crawler ni resuelve
# sitios dinámicos, protegidos o con autenticación.
# EN CLASE: revisar el texto cargado y señalar navegación, cookies o ruido a limpiar.
"""
from langchain_community.document_loaders import WebBaseLoader

url = "https://python.langchain.com/docs/introduction/"
documentos = WebBaseLoader(url).load()

print("Documentos:", len(documentos))
print("Fuente:", documentos[0].metadata.get("source"))
print(documentos[0].page_content[:300])
