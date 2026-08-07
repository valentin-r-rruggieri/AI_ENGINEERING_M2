"""Paso 8: la aplicacion completa. Todo lo anterior, ensamblado en un RAG que responde.

Pregunta -> FAISS (paso 5) -> filtro de metadata (paso 6) -> contexto -> LLM -> respuesta con fuentes.
"""
from comun import CHAT_MODEL, PREGUNTA, client
from paso06_filtros_metadata import buscar_con_filtro


def responder(pregunta: str, department: str, top_k: int = 3) -> str:
    resultados = buscar_con_filtro(pregunta, department, top_k=top_k)
    if not resultados:
        return "No encontre informacion autorizada para responder esa pregunta."

    contexto = "\n".join(f"[Fuente: {r['metadata']['source']}] {r['text']}" for r in resultados)

    respuesta = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Responde solo usando el contexto. Cita la fuente."},
            {"role": "user", "content": f"Pregunta: {pregunta}\n\nContexto:\n{contexto}"},
        ],
    )
    return respuesta.choices[0].message.content


if __name__ == "__main__":
    department = "IT"
    print(f"Pregunta: {PREGUNTA}  (usuario de {department})")
    print(f"Respuesta: {responder(PREGUNTA, department)}")
