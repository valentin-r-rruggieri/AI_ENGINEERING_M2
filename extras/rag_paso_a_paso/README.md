# RAG paso a paso (armado incremental)

Los mismos conceptos de `notebooks/AEM2L2_vector_stores` (E01-E10), pero armados
como una aplicacion real: cada paso corre, guarda su resultado en `estado/`, y el
paso siguiente lo retoma. Al final (`paso08_rag_final.py`) queda un mini-RAG
funcionando de punta a punta.

Usa el mismo dataset de FAQs que `extras/embeddings` (wifi/red, facturacion,
seguridad), con un departamento asignado a cada fuente para poder practicar
filtros de metadata.

## Los pasos

| Paso | Que hace | Concepto relacionado |
|---|---|---|
| `paso01_registro_vectorial.py` | Chunking + metadata (todavia sin embeddings) | E01 |
| `paso02_embeddings.py` | Genera el embedding de cada chunk | Clase 1 (embeddings) |
| `paso03_busqueda_exacta.py` | Busqueda lineal, comparando contra todos (k-NN exacto) | E03 |
| `paso04_metricas_normalizacion.py` | La trampa de la magnitud, y por que normalizar | E02 |
| `paso05_indice_faiss.py` | Reemplaza la busqueda lineal por un indice FAISS persistido | E04 |
| `paso06_filtros_metadata.py` | Filtra los resultados de FAISS por departamento | E05 |
| `paso07_recall_evaluacion.py` | Mide Recall@K comparando exacto vs FAISS | E06 |
| `paso08_rag_final.py` | Arma el contexto y le pregunta al LLM: la aplicacion completa | Pipeline RAG |

Temas mas avanzados (HNSW/IVF, benchmark de p50/p95, comparar FAISS contra
Chroma) quedan en las notebooks `E08`, `E09` y `E10` — no se repiten aca para
no duplicar contenido.

## Instalar

```bash
& ".\.venv\Scripts\python.exe" -m pip install -r ".\rag_paso_a_paso\requirements.txt"
```

La key se toma del `.env` que ya existe en la raiz del proyecto.

## Ejecutar, en orden

```bash
& ".\.venv\Scripts\python.exe" ".\rag_paso_a_paso\paso01_registro_vectorial.py"
& ".\.venv\Scripts\python.exe" ".\rag_paso_a_paso\paso02_embeddings.py"
& ".\.venv\Scripts\python.exe" ".\rag_paso_a_paso\paso03_busqueda_exacta.py"
& ".\.venv\Scripts\python.exe" ".\rag_paso_a_paso\paso04_metricas_normalizacion.py"
& ".\.venv\Scripts\python.exe" ".\rag_paso_a_paso\paso05_indice_faiss.py"
& ".\.venv\Scripts\python.exe" ".\rag_paso_a_paso\paso06_filtros_metadata.py"
& ".\.venv\Scripts\python.exe" ".\rag_paso_a_paso\paso07_recall_evaluacion.py"
& ".\.venv\Scripts\python.exe" ".\rag_paso_a_paso\paso08_rag_final.py"
```

Cada paso a partir del 02 necesita que el anterior ya haya corrido (usan lo
que quedo guardado en `estado/`). Si algo falla, se puede borrar `estado/` y
volver a correr desde `paso01`.
