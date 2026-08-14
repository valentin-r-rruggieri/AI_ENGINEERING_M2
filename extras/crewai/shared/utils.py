# Este modulo permite ejecutar los ejercicios de CrewAI aun sin instalar el framework. Cuando el
# paquete exista, cada archivo queda listo para reemplazar esta comprobacion por una Crew real.

# importlib.util detecta CrewAI sin producir errores de import.
import importlib.util

def run_lesson(topic: str) -> None:
    if importlib.util.find_spec("crewai") is None:
        print(f"{topic}: instala crewai con pip install crewai")
    else:
        print(f"{topic}: CrewAI listo para ejecutar agentes, tasks y crews")

# Resumen final: CrewAI estructura equipos de agentes, tareas y procesos de colaboracion.
