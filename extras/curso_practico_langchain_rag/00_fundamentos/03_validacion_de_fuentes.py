"""Fundamentos: validar una fuente antes de cargarla en un RAG.

Uso:
    python 03_validacion_de_fuentes.py

Lee el FAQ en UTF-8, comprueba que no está vacío, revisa encabezados Markdown,
detecta duplicados exactos de secciones y crea un Document trazable.
# GUÍA DOCENTE
# CUÁNDO USAR: antes de dividir e indexar una fuente real.
# DIFERENCIA: cargar no significa que los datos sirvan; validar detecta archivos
# vacíos, secciones faltantes y contenido repetido antes de pagar embeddings.
# EN CLASE: quitar temporalmente una sección y explicar por qué debe fallar.
"""

from collections import Counter
from datetime import date
from pathlib import Path
import re

from langchain_core.documents import Document


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    while current.parent != current:
        if (current / ".env.example").is_file() and (current / "data").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("No se encontró la raíz del curso.")


def main() -> None:
    root = find_project_root(Path.cwd())
    source = root / "data" / "faq_empresa_saas.txt"
    text = source.read_text(encoding="utf-8").strip()
    headings = re.findall(r"^##\s+(.+)$", text, flags=re.MULTILINE)
    duplicates = [heading for heading, count in Counter(headings).items() if count > 1]

    assert text, "La fuente está vacía. No se debe indexar."
    assert len(headings) >= 3, "Se esperaban al menos tres secciones Markdown."
    assert not duplicates, f"Secciones duplicadas: {duplicates}"

    document = Document(
        page_content=text,
        metadata={
            "source": source.name,
            "source_path": str(source),
            "version": "2026-08",
            "indexed_on": date.today().isoformat(),
            "sections": len(headings),
        },
    )

    print("FUENTE:", source.name)
    print("CARACTERES:", len(text))
    print("SECCIONES:", len(headings))
    for heading in headings:
        print("-", heading)
    print("\nMETADATA DEL DOCUMENT:", document.metadata)
    print("\nOK: fuente válida para la siguiente etapa del pipeline.")


if __name__ == "__main__":
    main()
