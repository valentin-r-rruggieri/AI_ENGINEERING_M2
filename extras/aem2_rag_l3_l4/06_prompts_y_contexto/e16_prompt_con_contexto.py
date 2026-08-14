# Este archivo arma un prompt aumentado sin llamar todavia a un modelo. Recupera contexto, lo
# inserta en ChatPromptTemplate y deja visible el mensaje final que recibiria el LLM. Al
# ejecutarlo se aprende a separar instrucciones, evidencia documental y pregunta del usuario.
# sys deja disponible el paquete shared durante la ejecucion directa.
import sys
# Path localiza la raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ChatPromptTemplate arma mensajes parametrizados para un modelo de chat.
from langchain_core.prompts import ChatPromptTemplate

# DOCUMENTS aporta el corpus sobre el que se busca contexto.
from shared.dataset import DOCUMENTS
# lexical_search recupera los documentos que se insertaran en el prompt.
from shared.utils import lexical_search

# Recuperar antes de crear el prompt evita mandar el corpus completo al modelo.
question = "Cuantos dias de vacaciones tengo?"
context = "\n".join(doc.page_content for doc in lexical_search(question, DOCUMENTS, k=2))

# La instruccion exige grounding: si la evidencia no alcanza, el modelo debe reconocerlo.
prompt = ChatPromptTemplate.from_template(
    "Responde solo con el CONTEXTO. Si no alcanza, di que no tienes evidencia.\n\nCONTEXTO:\n{context}\n\nPREGUNTA: {question}"
)
print(prompt.invoke({"context": context, "question": question}).to_string())
print("Pregunta y contexto se pasan por separado para que los documentos se traten como datos.")

# Resumen final: un prompt RAG debe delimitar contexto, pregunta e instrucciones claramente.
# El modelo debe recibir una regla de abstencion cuando la evidencia no alcanza para responder.
