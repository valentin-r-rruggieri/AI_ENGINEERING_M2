"""Fundamentos: encontrar el proyecto y validar configuración sin exponer secretos.

Uso:
    python 02_entorno_dotenv_y_configuracion.py

El script busca la raíz del curso, carga .env si existe y muestra únicamente si
cada variable está configurada. Nunca imprime valores de API keys.
"""

from pathlib import Path
import os

from dotenv import load_dotenv


def find_project_root(start: Path) -> Path:
    """Busca la carpeta que contiene .env.example y data/."""
    current = start.resolve()
    while current.parent != current:
        if (current / ".env.example").is_file() and (current / "data").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("No se encontró la raíz de curso_practico_langchain_rag.")


def configured(name: str) -> bool:
    """Indica presencia de una variable sin revelar su contenido."""
    return bool(os.getenv(name, "").strip())


def main() -> None:
    root = find_project_root(Path.cwd())
    load_dotenv(root / ".env", override=False)

    groups = {
        "OpenAI": ["OPENAI_API_KEY", "OPENAI_CHAT_MODEL", "OPENAI_EMBEDDING_MODEL"],
        "Pinecone": ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "PINECONE_NAMESPACE"],
        "LangSmith": ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"],
    }

    print("RAÍZ DEL CURSO:", root)
    for group, names in groups.items():
        print(f"\n{group}:")
        for name in names:
            print(f"- {name}: {'configurada' if configured(name) else 'no configurada'}")

    print("\nOK: las credenciales se validaron sin mostrarse.")


if __name__ == "__main__":
    main()
