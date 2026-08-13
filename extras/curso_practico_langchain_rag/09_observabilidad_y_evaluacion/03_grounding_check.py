"""Grounding significa que la respuesta se apoya en contexto recuperado.
El control básico es mostrar respuesta y fuentes juntas para revisión humana.
"""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

documentos = [Document(page_content="El soporte se contacta por email.", metadata={"source": "faq_empresa_saas.txt"})]
contexto = "\n".join(documento.page_content for documento in documentos)
prompt = ChatPromptTemplate.from_template("Usa solamente este contexto:\n{context}\n\nPregunta: {question}")

print(prompt.format(context=contexto, question="¿Cómo contacto soporte?"))
print("Fuente:", documentos[0].metadata["source"])
