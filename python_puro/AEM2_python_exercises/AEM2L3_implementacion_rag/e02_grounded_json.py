"""E02: construir y validar el contrato de respuesta RAG."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import BaseModel, Field

class Answer(BaseModel):
    user_question:str=Field(min_length=1)
    system_answer:str=Field(min_length=1)
    chunks_related:list[dict]=Field(min_length=1,max_length=5)

def build_response(question, answer, chunks):
    return Answer(user_question=question,system_answer=answer,chunks_related=chunks)

response=build_response("¿Cómo recupero mi contraseña?","Usá el enlace enviado al correo.",[{"chunk_id":"faq-001","source":"faq"}])
assert set(response.model_dump())=={"user_question","system_answer","chunks_related"}
print(response.model_dump_json(indent=2))

