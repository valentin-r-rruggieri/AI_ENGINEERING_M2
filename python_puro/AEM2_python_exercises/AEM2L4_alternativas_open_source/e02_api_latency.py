"""E02: medir latencia de un lote de embeddings de API."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import embed_openai, timed, write_json

# Usamos un lote pequeño y representativo; la medición real debe repetirse varias veces.
texts=["recuperar contraseña","solicitar vacaciones","auditoría de usuarios"]
# timed devuelve el resultado y el tiempo transcurrido en milisegundos.
_,latency=timed(embed_openai,texts)
# Registrar cantidad de textos evita comparar latencias de cargas diferentes.
result={"texts":len(texts),"latency_ms":round(latency,2)}
write_json(Path(__file__).parent/"data"/"api_latency.json",result); print(result)
