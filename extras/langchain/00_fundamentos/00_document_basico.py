# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""Fundamentos: crear y validar un Document de LangChain.

Uso:
    python 00_document_basico.py

Un Document es el contrato común que comparten loaders, splitters, vector stores
y retrievers. `page_content` contiene el texto recuperable; `metadata` conserva
información para citar, filtrar y auditar la fuente.
# GUÍA DOCENTE
# CUÁNDO USAR: al iniciar cualquier RAG, porque Document es el contrato común
# entre loaders, splitters, vector stores y retrievers.
# DIFERENCIA: page_content es el texto recuperable; metadata describe origen,
# página, sección o permisos y permite citar/filtrar sin contaminar el texto.
# EN CLASE: cambiar el texto y sumar metadata; observar que ambos viajan juntos.
"""

# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.documents import Document


def main() -> None:
    document = Document(
        page_content="Las vacaciones se solicitan desde Mi perfil > Ausencias.",
        metadata={
            "source": "faq_empresa_saas.txt",
            "section": "Licencias",
            "language": "es",
        },
    )

    # Estas verificaciones son una prueba mínima del contrato Document.
    assert isinstance(document, Document)
    assert document.page_content.strip()
    assert {"source", "section"}.issubset(document.metadata)

    print("TIPO:", type(document).__name__)
    print("\nPAGE_CONTENT:")
    print(document.page_content)
    print("\nMETADATA:")
    for key, value in document.metadata.items():
        print(f"- {key}: {value}")
    print("\nOK: Document válido con contenido y trazabilidad.")


if __name__ == "__main__":
    main()

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
