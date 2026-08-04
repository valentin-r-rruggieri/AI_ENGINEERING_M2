"""E01: matriz reproducible para una decisión de arquitectura."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import write_json

options={"api":{"quality":4,"latency":4,"privacy":2,"operations":5,"cost":3},"local":{"quality":3,"latency":3,"privacy":5,"operations":2,"cost":4}}
weights={"quality":0.30,"latency":0.20,"privacy":0.25,"operations":0.15,"cost":0.10}
result={name:round(sum(scores[k]*weights[k] for k in weights),2) for name,scores in options.items()}
write_json(Path(__file__).parent/"data"/"decision.json",result); print(result)

