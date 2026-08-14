# Este archivo introduce pruebas de rutas del grafo con LangGraph. Es un paso corto dentro del curso y permite estudiar
# una sola idea antes de combinarla con memoria, RAG o multiagente. La ejecucion es didactica y segura.

# sys permite importar los helpers compartidos desde un ejercicio ejecutado directamente.
import sys
# Path calcula la raiz del curso a partir de la carpeta actual.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# run_lesson centraliza la comprobacion local o de dependencias del framework.
from shared.utils import run_lesson

# El concepto se mantiene separado para que el resultado sea facil de modificar y observar.
run_lesson("pruebas de rutas del grafo")

# Resumen final: pruebas de rutas del grafo es una pieza concreta de LangGraph que se combina despues con el resto del agente.
