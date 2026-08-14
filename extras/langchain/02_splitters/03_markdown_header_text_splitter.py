# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""MarkdownHeaderTextSplitter: transforma encabezados Markdown en metadata.
Preserva la sección (h1/h2) y permite recuperar o citar por tema.
# GUÍA DOCENTE
# CUÁNDO USAR: Markdown bien organizado con títulos y subtítulos.
# DIFERENCIA: conserva la jerarquía como metadata h1/h2; un splitter recursivo
# preserva tamaño, pero no sabe qué sección está leyendo.
# EN CLASE: filtrar o citar usando el encabezado guardado en metadata.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown = "# Producto\nInformación general.\n## Facturación\nLas facturas son mensuales.\n## Soporte\nContacta por email."
documentos = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2")]).split_text(markdown)

for documento in documentos:
    print(documento.metadata, "=>", documento.page_content)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
