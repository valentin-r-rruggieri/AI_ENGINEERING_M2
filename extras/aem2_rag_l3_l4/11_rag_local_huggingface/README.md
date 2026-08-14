# RAG local con Hugging Face

Esta carpeta es un recorrido separado para construir un RAG que corre en la maquina local. Los archivos estan ordenados para ejecutarse de arriba hacia abajo, aunque cada uno tambien funciona por separado.

```bash
cd /Users/valentin/AI_ENGINEERING_M2/extras/aem2_rag_l3_l4
pip install -r requirements.txt
python3 11_rag_local_huggingface/00_revisar_entorno_local.py
```

Los modelos se descargan la primera vez desde Hugging Face y luego quedan en la cache local. El embedding recomendado para estos ejemplos es `sentence-transformers/all-MiniLM-L6-v2`; para generacion local se propone `google/flan-t5-small`.

Recorrido: entorno -> embeddings -> chunks -> FAISS -> retrievers -> filtros -> persistencia -> respuesta con evidencia -> LLM local -> LCEL -> evaluacion.
