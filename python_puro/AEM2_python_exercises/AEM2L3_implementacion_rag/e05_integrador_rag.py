"""E05: RAG mínimo con retrieval, grounding y Responses API."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openai import OpenAI
from common import embed_openai, require_openai_key, setting, split_words, top_k, write_json

root=Path(__file__).parent; question="¿Cómo solicito vacaciones?"
chunks=split_words((root/"data"/"faq.txt").read_text(encoding="utf-8"),35,8,"faq")
vectors=embed_openai([c.content for c in chunks]); results=top_k(embed_openai([question])[0],chunks,vectors,3)
context="\n".join(f"[{r['chunk']['chunk_id']}] {r['chunk']['content']}" for r in results)
prompt=f"Respondé solo con este contexto. Si falta evidencia, decí que no hay información suficiente.\n{context}\nPregunta: {question}"
answer=OpenAI(api_key=require_openai_key()).responses.create(model=setting("AEM2_GENERATION_MODEL","gpt-5.6-luna"),input=prompt).output_text
payload={"user_question":question,"system_answer":answer,"chunks_related":[r["chunk"] for r in results]}
write_json(root/"data"/"sample_response.json",payload); print(payload)

