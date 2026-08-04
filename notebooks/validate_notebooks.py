"""Valida estructura pedagógica y sintaxis de los notebooks AEM2."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_NOTEBOOKS = ROOT.parent / "proyecto_integrador" / "notebooks"
REQUIRED_HEADINGS = (
    "## Referencia visual y conceptual",
    "## Antes de ejecutar",
    "## Práctica guiada",
    "## Resultado esperado",
    "## Errores frecuentes",
    "## Mini desafío",
    "## Cierre",
)


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


def validate(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError(f"{path}: nbformat inválido")
    cells = notebook.get("cells", [])
    markdown = "\n".join(source(cell) for cell in cells if cell.get("cell_type") == "markdown")
    missing = [heading for heading in REQUIRED_HEADINGS if heading not in markdown]
    model_mental_markers = ("## Modelo mental", "## 1.", "## Arquitectura mínima")
    if not any(marker in markdown for marker in model_mental_markers):
        missing.append("## Modelo mental")
    if missing:
        raise ValueError(f"{path}: faltan secciones {missing}")
    code_cells = [source(cell) for cell in cells if cell.get("cell_type") == "code"]
    if not code_cells:
        raise ValueError(f"{path}: falta práctica ejecutable")
    for number, code in enumerate(code_cells, 1):
        compile(code, f"{path.name}:cell-{number}", "exec")


def main() -> None:
    paths = sorted(ROOT.glob("AEM2L*/*.ipynb")) + sorted(PROJECT_NOTEBOOKS.glob("*.ipynb"))
    expected = 49
    if len(paths) != expected:
        raise ValueError(f"Se esperaban {expected} notebooks; se encontraron {len(paths)}.")
    for path in paths:
        validate(path)
    print(f"OK: {len(paths)} notebooks con teoría, práctica y sintaxis válida.")


if __name__ == "__main__":
    main()
