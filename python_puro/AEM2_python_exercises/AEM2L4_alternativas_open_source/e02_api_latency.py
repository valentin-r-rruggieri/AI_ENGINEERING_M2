"""E02: medir latencia de un lote de embeddings de API."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import embed_openai, timed, write_json

texts=["recuperar contraseña","solicitar vacaciones","auditoría de usuarios"]
_,latency=timed(embed_openai,texts)
result={"texts":len(texts),"latency_ms":round(latency,2)}
write_json(Path(__file__).parent/"data"/"api_latency.json",result); print(result)

