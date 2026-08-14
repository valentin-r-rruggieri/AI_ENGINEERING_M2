# Este modulo evita llamadas cloud accidentales. Los ejercicios de Agents SDK comprueban primero
# que el paquete y OPENAI_API_KEY existan, y si no explican como configurar el entorno.

# importlib.util detecta el SDK sin intentar importarlo cuando falta.
import importlib.util
# os permite leer credenciales desde el entorno.
import os

def run_lesson(topic: str) -> None:
    if importlib.util.find_spec("agents") is None:
        print(f"{topic}: instala openai-agents con pip install openai-agents")
    elif not os.getenv("OPENAI_API_KEY"):
        print(f"{topic}: configura OPENAI_API_KEY en .env antes de ejecutar una llamada real")
    else:
        print(f"{topic}: SDK y clave listos para ejecutar el ejemplo real")

# Resumen final: los agentes cloud requieren SDK y credenciales fuera del repositorio.
