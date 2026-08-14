# Curso Practico LlamaIndex RAG

Curso paralelo a `langchain`, organizado en las mismas 11 familias. Usa `Document`, `Node`, `VectorStoreIndex`, `Retriever`, `QueryEngine`, `ResponseSynthesizer` y `Settings` de LlamaIndex.

Los scripts son lineales, con introduccion teorica, comentarios de imports y resumen final. Los ejemplos base usan `MockEmbedding` y `MockLLM`, por lo que no requieren API key.

```bash
cd /Users/valentin/AI_ENGINEERING_M2/extras/llamaindex
pip install -r requirements.txt
python3 00_fundamentos/00_document_basico.py
```

Instala `requirements-optional.txt` para Hugging Face, Chroma y FAISS. Usa `.env` solo para los ejercicios de OpenAI.
