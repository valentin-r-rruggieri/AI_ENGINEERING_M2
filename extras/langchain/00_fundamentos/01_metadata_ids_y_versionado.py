# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""Fundamentos: metadata, IDs determinísticos y versionado.

Uso:
    python 01_metadata_ids_y_versionado.py

Un ID determinístico permite volver a indexar sin duplicar lógicamente el mismo
chunk. El hash se calcula sobre los campos que definen la identidad del contenido.
Cambiar el texto o la versión cambia el ID y obliga a tratarlo como otra revisión.
# GUÍA DOCENTE
# CUÁNDO USAR: cuando el corpus se actualiza y se necesita evitar duplicados.
# DIFERENCIA: metadata explica el documento; un ID estable identifica la misma
# unidad entre indexaciones. Cambiar contenido o versión debe producir un ID nuevo.
# EN CLASE: mostrar que el hash no es semántico: un cambio pequeño cambia el ID.
"""

# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from hashlib import sha256
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.documents import Document


def deterministic_id(*parts: str) -> str:
    """Devuelve un ID estable de 16 caracteres a partir de datos normalizados."""
    normalized = "|".join(part.strip().lower() for part in parts)
    return sha256(normalized.encode("utf-8")).hexdigest()[:16]


def make_document(content: str, version: str) -> Document:
    source = "faq_empresa_saas.txt"
    section = "Licencias"
    chunk_id = deterministic_id(source, section, version, content)
    return Document(
        page_content=content,
        metadata={
            "id": chunk_id,
            "source": source,
            "section": section,
            "version": version,
        },
    )


def main() -> None:
    original = make_document(
        "Las vacaciones se solicitan desde Mi perfil > Ausencias.", "2026-08"
    )
    same_content = make_document(
        "Las vacaciones se solicitan desde Mi perfil > Ausencias.", "2026-08"
    )
    revised = make_document(
        "Las vacaciones se solicitan desde Mi perfil > Ausencias > Nueva solicitud.",
        "2026-09",
    )

    assert original.metadata["id"] == same_content.metadata["id"]
    assert original.metadata["id"] != revised.metadata["id"]

    print("ID original: ", original.metadata["id"])
    print("ID repetido: ", same_content.metadata["id"])
    print("ID revisado: ", revised.metadata["id"])
    print("\nMetadata original:")
    for key, value in original.metadata.items():
        print(f"- {key}: {value}")
    print("\nOK: el ID es estable para el mismo contenido y cambia con la revisión.")


if __name__ == "__main__":
    main()

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
