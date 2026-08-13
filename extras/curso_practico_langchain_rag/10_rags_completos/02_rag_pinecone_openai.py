"""RAG Pinecone: consulta un índice cloud de operaciones.
El índice existente debe haberse cargado con data/rag_pinecone; no crea ni elimina recursos.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

curso = Path(__file__).resolve().parents[1]
load_dotenv(curso / ".env")
documentos = TextLoader(str(curso / "data" / "rag_pinecone" / "manual_operaciones_cloud.txt"), encoding="utf-8").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60).split_documents(documentos)
print("Corpus esperado en Pinecone:", len(chunks), "chunks de operaciones cloud.")
if not all(os.getenv(variable) for variable in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]):
    print("Faltan variables de OpenAI o Pinecone en .env")
else:
    try:
        from pinecone import Pinecone
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_pinecone import PineconeVectorStore
        from langchain_core.prompts import ChatPromptTemplate
        embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        indice = pinecone.Index(os.environ["PINECONE_INDEX_NAME"])
        store = PineconeVectorStore(index=indice, embedding=embeddings, namespace=os.getenv("PINECONE_NAMESPACE", ""))
        recuperados = store.similarity_search("¿Qué se debe hacer ante una alerta de latencia?", k=3)
        contexto = "\n\n".join(documento.page_content for documento in recuperados)
        prompt = ChatPromptTemplate.from_template("Usa solo el contexto. Sin evidencia di que no la tienes.\nContexto: {context}\nPregunta: {question}")
        respuesta = (prompt | ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)).invoke({"context": contexto, "question": "¿Qué se debe hacer ante una alerta de latencia?"})
        print(respuesta.content)
    except ImportError:
        print("Instala Pinecone: pip install pinecone langchain-pinecone")
