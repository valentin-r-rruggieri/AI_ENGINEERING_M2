# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""Fundamentos: encontrar el proyecto y validar configuración sin exponer secretos.

Uso:
    python 02_entorno_dotenv_y_configuracion.py

El script busca la raíz del curso, carga .env si existe y muestra únicamente si
cada variable está configurada. Nunca imprime valores de API keys.
# GUÍA DOCENTE
# CUÁNDO USAR: antes de cualquier llamada a OpenAI, Pinecone o LangSmith.
# DIFERENCIA: .env guarda secretos fuera del código; .env.example solo documenta
# los nombres de variables y sí se puede compartir.
# EN CLASE: mostrar estados configurada/ausente, nunca imprimir claves.
"""

# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
import os

# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from dotenv import load_dotenv


def find_project_root(start: Path) -> Path:
    """Busca la carpeta que contiene .env.example y data/."""
    current = start.resolve()
    while current.parent != current:
        if (current / ".env.example").is_file() and (current / "data").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("No se encontró la raíz de langchain.")


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

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
