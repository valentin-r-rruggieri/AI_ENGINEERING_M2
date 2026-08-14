# AEM2 RAG L3/L4

Curso practico de RAG organizado por concepto, con 32 ejercicios Python independientes. Cada archivo es lineal, sin `main()`, y explica la teoria con comentarios junto a la parte de LangChain que la aplica.

## Preparacion

```bash
cd /Users/valentin/AI_ENGINEERING_M2/extras/aem2_rag_l3_l4
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`OPENAI_API_KEY` solo es necesaria para los ejercicios cloud. Los ejercicios de corpus, splitters, prompts, metricas y arquitectura se pueden ejecutar sin clave.

## Ejecucion

Cada script se ejecuta directamente desde cualquier ubicacion:

```bash
python3 00_fundamentos_rag/e01_ingestion_chunking_metadata.py
python3 05_retrievers/e14_mmr_retriever.py
python3 09_observabilidad_y_metricas/e27_recall_precision_mrr.py
```

Los ejercicios `e06` y `e19` usan OpenAI. Los ejercicios `e07`, `e10` y `e20` requieren dependencias opcionales locales (`langchain-huggingface`, `sentence-transformers` y/o `faiss-cpu`) y explican como instalarlas si faltan.

La carpeta [11_rag_local_huggingface](/Users/valentin/AI_ENGINEERING_M2/extras/aem2_rag_l3_l4/11_rag_local_huggingface) contiene un recorrido exclusivo para construir un RAG local paso a paso con Hugging Face, FAISS y Transformers.

Consulta `catalogo_scripts.json` para el inventario completo.
