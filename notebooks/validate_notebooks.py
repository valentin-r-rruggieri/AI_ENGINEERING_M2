"""Validate the technical structure and Python syntax of AEM2 notebooks."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_NOTEBOOKS = ROOT.parent / "proyecto_integrador" / "notebooks"
FORBIDDEN_HEADING_PREFIXES = (
    "referencia visual",
    "fuente de inspiracion",
    "objetivos",
    "antes de ejecutar",
    "resultado esperado",
    "mini desafio",
    "cierre",
    "ejercicios practicos",
)
THEORY_MARKERS = (
    "## conceptos clave",
    "## 1.",
    "## arquitectura m",
    "## componentes verificables",
    "## problema real",
)


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


def normalized(value: str) -> str:
    return unicodedata.normalize("NFKD", value.casefold()).encode("ascii", "ignore").decode()


def validate(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError(f"{path}: nbformat invalido")

    cells = notebook.get("cells", [])
    markdown = "\n".join(source(cell) for cell in cells if cell.get("cell_type") == "markdown")
    normalized_markdown = normalized(markdown)
    headings = [
        normalized(line.lstrip("#").strip())
        for line in markdown.splitlines()
        if line.startswith("#")
    ]
    forbidden = [
        heading
        for heading in headings
        if any(heading.startswith(prefix) for prefix in FORBIDDEN_HEADING_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"{path}: contiene secciones eliminadas: {forbidden}")
    if not any(marker in normalized_markdown for marker in THEORY_MARKERS):
        raise ValueError(f"{path}: falta contenido teorico")

    code_cells = [source(cell) for cell in cells if cell.get("cell_type") == "code"]
    if not code_cells:
        raise ValueError(f"{path}: falta codigo ejecutable")
    for number, code in enumerate(code_cells, 1):
        if not code.strip():
            raise ValueError(f"{path}: celda de codigo vacia ({number})")
        if not code.lstrip().startswith("# Ejemplo ejecutable:"):
            raise ValueError(f"{path}: falta comentario descriptivo en celda {number}")
        compile(code, f"{path.name}:cell-{number}", "exec")


def main() -> None:
    paths = sorted(ROOT.glob("AEM2L*/*.ipynb")) + sorted(PROJECT_NOTEBOOKS.glob("*.ipynb"))
    expected = 49
    if len(paths) != expected:
        raise ValueError(f"Se esperaban {expected} notebooks; se encontraron {len(paths)}.")
    for path in paths:
        validate(path)
    print(f"OK: {len(paths)} notebooks con teoria, codigo y sintaxis valida.")


if __name__ == "__main__":
    main()
