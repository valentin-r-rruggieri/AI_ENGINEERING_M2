"""Paso 8: la aplicacion completa. Todo lo anterior, ensamblado en un RAG que responde.

Pregunta -> FAISS (paso 5) -> filtro de metadata (paso 6) -> contexto -> LLM -> respuesta con fuentes.
Docs: https://python.langchain.com/docs/concepts/prompt_templates/
"""
from langchain_core.prompts import ChatPromptTemplate

from comun import PREGUNTA, llm
from paso06_filtros_metadata import buscar_con_filtro


def responder(pregunta: str, department: str, top_k: int = 3) -> str:
    resultados = buscar_con_filtro(pregunta, department, top_k=top_k)
    if not resultados:
        return "No encontre informacion autorizada para responder esa pregunta."

    contexto = "\n".join(f"[Fuente: {doc.metadata['source']}] {doc.page_content}" for doc, _ in resultados)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Responde solo usando el contexto. Cita la fuente."),
            ("human", "Pregunta: {pregunta}\n\nContexto:\n{contexto}"),
        ]
    )
    respuesta = llm.invoke(prompt.invoke({"pregunta": pregunta, "contexto": contexto}))
    return respuesta.content


if __name__ == "__main__":
    department = "IT"
    print(f"Pregunta: {PREGUNTA}  (usuario de {department})")
    print(f"Respuesta: {responder(PREGUNTA, department)}")
