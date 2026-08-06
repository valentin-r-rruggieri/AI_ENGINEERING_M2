"""E05: consolidar benchmark y producir una recomendación explicable."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_json, write_json

root=Path(__file__).parent; matrix=read_json(root/"data"/"decision.json")
# max elige el mayor puntaje ponderado calculado en el ejercicio anterior.
winner=max(matrix,key=matrix.get)
# La advertencia recuerda que una matriz no sustituye pruebas con datos representativos.
report={"winner":winner,"scores":matrix,"warning":"Validar con golden cases y datos representativos antes de producción."}
write_json(root/"data"/"benchmark_report.json",report); print(report)
