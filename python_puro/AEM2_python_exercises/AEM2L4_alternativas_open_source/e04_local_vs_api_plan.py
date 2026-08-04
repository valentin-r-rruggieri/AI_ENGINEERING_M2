"""E04: registrar supuestos para comparar API y modelo local."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import write_json

report={"api":{"network":True,"gpu":False,"data_leaves_environment":True},"local":{"network":False,"gpu":"depende del modelo","data_leaves_environment":False},"next_step":"ejecutar el mismo golden set con ambos modelos"}
write_json(Path(__file__).parent/"data"/"comparison_plan.json",report); print(report)

