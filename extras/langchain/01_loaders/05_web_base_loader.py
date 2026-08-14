# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""WebBaseLoader: carga texto de una página HTML pública.
Úsalo para una URL puntual y revisa permisos, limpieza y contenido duplicado.
# GUÍA DOCENTE
# CUÁNDO USAR: una página pública y estable que se puede descargar legalmente.
# DIFERENCIA: recupera HTML convertido a texto; no sustituye un crawler ni resuelve
# sitios dinámicos, protegidos o con autenticación.
# EN CLASE: revisar el texto cargado y señalar navegación, cookies o ruido a limpiar.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.document_loaders import WebBaseLoader

url = "https://python.langchain.com/docs/introduction/"
documentos = WebBaseLoader(url).load()

print("Documentos:", len(documentos))
print("Fuente:", documentos[0].metadata.get("source"))
print(documentos[0].page_content[:300])

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
