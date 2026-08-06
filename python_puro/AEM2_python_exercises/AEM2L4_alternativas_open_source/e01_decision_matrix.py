"""E01: matriz reproducible para una decisión de arquitectura."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import write_json

# Cada alternativa recibe puntajes de 1 a 5: los datos deben provenir de evidencia.
options={"api":{"quality":4,"latency":4,"privacy":2,"operations":5,"cost":3},"local":{"quality":3,"latency":3,"privacy":5,"operations":2,"cost":4}}
# Los pesos hacen explícita la prioridad del caso de uso; deben sumar 1.0.
weights={"quality":0.30,"latency":0.20,"privacy":0.25,"operations":0.15,"cost":0.10}
# Un puntaje ponderado no decide por sí solo: sirve para justificar la conversación técnica.
result={name:round(sum(scores[k]*weights[k] for k in weights),2) for name,scores in options.items()}
write_json(Path(__file__).parent/"data"/"decision.json",result); print(result)
