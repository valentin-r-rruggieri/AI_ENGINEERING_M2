# RAG paso a paso con LangChain (armado incremental)

Misma progresión que [`extras/rag_paso_a_paso`](../rag_paso_a_paso/README.md), pero hecha
enteramente con herramientas de LangChain en vez de OpenAI SDK + FAISS crudo:
`RecursiveCharacterTextSplitter` para chunking, `Document` para el registro,
`InMemoryVectorStore` para la búsqueda exacta, `FAISS` (`langchain_community`)
para el índice persistente, y `ChatPromptTemplate` + `ChatOpenAI` para la
respuesta final. Ningún paso usa `numpy` para calcular similitud — solo se usa
`numpy` una vez, en el paso 4, para mostrar a propósito qué pasaría si **no**
se usara la métrica que LangChain calcula por dentro.

Usa el mismo dataset de FAQs que `extras/embeddings` (wifi/red, facturación,
seguridad), con un departamento asignado a cada fuente para poder practicar
filtros de metadata.

## Los pasos

| Paso | Que hace | Herramienta de LangChain |
|---|---|---|
| `paso01_documentos.py` | Chunking + metadata (todavía sin embeddings) | `RecursiveCharacterTextSplitter`, `Document` |
| `paso02_embeddings.py` | Genera el embedding de cada Document | `OpenAIEmbeddings.embed_documents()` |
| `paso03_vectorstore_memoria.py` | Búsqueda "exacta" en memoria (coseno) | `InMemoryVectorStore` |
| `paso04_metricas_normalizacion.py` | La trampa de la magnitud, con un `Embeddings` de juguete | `Embeddings` (clase base), `InMemoryVectorStore` |
| `paso05_vectorstore_faiss.py` | Índice FAISS persistido (reemplaza al paso 3) | `FAISS.from_documents` / `save_local` / `load_local` |
| `paso06_filtros_metadata.py` | Filtra por departamento dentro de la búsqueda | `similarity_search_with_score(..., filter=...)` |
| `paso07_recall_evaluacion.py` | Recall@K comparando `InMemoryVectorStore` vs `FAISS` | (evaluación, sin herramienta específica) |
| `paso08_rag_final.py` | Arma el contexto y le pregunta al LLM: la aplicación completa | `ChatPromptTemplate`, `ChatOpenAI` |

## Una diferencia importante entre el paso 3 y el paso 5

`InMemoryVectorStore` calcula **similitud coseno** (más alto es mejor).
`FAISS` de LangChain, por default, calcula **distancia L2** (más bajo es mejor).
Vas a ver que los números cambian de escala y de sentido entre `paso03` y
`paso05`, aunque el *orden* de los resultados sea el mismo — es la misma idea
de la diapositiva de "métricas y normalización", pero encontrada en la
práctica en vez de en una diapositiva.

## Nota sobre el warning de deprecación

Al correr `paso05` en adelante vas a ver:

```
DeprecationWarning: `langchain-community` is being sunset...
```

Es esperable: LangChain está migrando sus integraciones a paquetes propios,
pero al momento de escribir esto no existe todavía un paquete standalone
funcional para FAISS (`langchain-faiss` en PyPI está vacío). `langchain_community.vectorstores.FAISS`
sigue siendo la forma correcta y funcional de usar FAISS desde LangChain hoy.

## Instalar

```bash
& ".\.venv\Scripts\python.exe" -m pip install -r ".\rag_paso_a_paso_langchain\requirements.txt"
```

La key se toma del `.env` que ya existe en la raíz del proyecto.

## Ejecutar, en orden

```bash
& ".\.venv\Scripts\python.exe" ".\rag_paso_a_paso_langchain\paso01_documentos.py"
& ".\.venv\Scripts\python.exe" ".\rag_paso_a_paso_langchain\paso02_embeddings.py"
& ".\.venv\Scripts\python.exe" ".\rag_paso_a_paso_langchain\paso03_vectorstore_memoria.py"
& ".\.venv\Scripts\python.exe" ".\rag_paso_a_paso_langchain\paso04_metricas_normalizacion.py"
& ".\.venv\Scripts\python.exe" ".\rag_paso_a_paso_langchain\paso05_vectorstore_faiss.py"
& ".\.venv\Scripts\python.exe" ".\rag_paso_a_paso_langchain\paso06_filtros_metadata.py"
& ".\.venv\Scripts\python.exe" ".\rag_paso_a_paso_langchain\paso07_recall_evaluacion.py"
& ".\.venv\Scripts\python.exe" ".\rag_paso_a_paso_langchain\paso08_rag_final.py"
```

Cada paso a partir del 03 necesita que el paso 1 ya haya corrido (usan lo que
quedó guardado en `estado/`). Si algo falla, se puede borrar `estado/` y
volver a correr desde `paso01`.
