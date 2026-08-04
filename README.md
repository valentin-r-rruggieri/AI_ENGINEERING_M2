# AEM2 — AI Engineering Module 2: RAG

Material didáctico para aprender a construir sistemas de **Retrieval-Augmented Generation (RAG)** desde los fundamentos hasta una implementación completa. El módulo está pensado para que una persona estudiante pueda avanzar de manera autónoma: primero entiende el concepto, luego lo prueba en un notebook y finalmente lo implementa con Python.

El caso transversal es una base de conocimiento de soporte y RR.HH. Esto permite trabajar con preguntas reales: políticas, vacaciones, accesos, contraseñas y procedimientos internos.

## Qué se aprende

Al completar el módulo, la persona estudiante podrá:

- Explicar qué representa un embedding y cuándo usar similitud coseno.
- Fragmentar documentos con tamaño, solapamiento y metadata trazable.
- Construir búsqueda vectorial exacta y aproximada con FAISS.
- Comparar índices Flat, HNSW e IVF usando Recall@K y latencia.
- Implementar un flujo RAG con grounding, recuperación, reranking y caché.
- Validar una respuesta estructurada y abstenerse cuando la evidencia no alcanza.
- Elegir entre API, modelos locales o una solución híbrida según costo, privacidad y rendimiento.
- Entregar un asistente FAQ con Chroma y FAISS comparables.

## Mapa del módulo

| Bloque | Pregunta que responde | Resultado práctico |
| --- | --- | --- |
| L1 — Embeddings y fragmentación | ¿Cómo convierte una máquina el significado del texto en números útiles? | Un recuperador semántico con chunks y golden cases. |
| L2 — Bases vectoriales | ¿Cómo se guardan y buscan miles de vectores de forma eficiente? | Índices exactos y ANN comparados con métricas. |
| L3 — Implementación de RAG | ¿Cómo se transforma una búsqueda en una respuesta fundamentada? | Un pipeline RAG trazable con salida JSON. |
| L4 — Alternativas y ecosistema | ¿Qué arquitectura conviene para cada restricción real? | Una matriz de decisión y un ADR con evidencia. |
| Proyecto integrador | ¿Cómo se une todo en una aplicación usable? | Un FAQ RAG de RR.HH. con Chroma y FAISS. |

## Orden recomendado de estudio

Seguí este recorrido. Cada clase comienza con un notebook `00`, que funciona como una clase completa para estudiante: teoría, gráficos, tablas, predicciones, práctica y desafío. Después continuá con los ejercicios `E01` a `E10`.

1. **L1:** [clase de embeddings](notebooks/AEM2L1_embeddings_fragmentacion/00_clase_teoria_y_practica_embeddings.ipynb) y `E01`–`E10`.
2. **L2:** [clase de vector stores](notebooks/AEM2L2_vector_stores/00_clase_teoria_y_practica_vector_stores.ipynb) y `E01`–`E10`.
3. **L3:** [clase de RAG](notebooks/AEM2L3_implementacion_rag/00_clase_teoria_y_practica_rag.ipynb) y `E01`–`E10`.
4. **L4:** [clase de ecosistema](notebooks/AEM2L4_alternativas_open_source/00_clase_teoria_y_practica_ecosistema.ipynb) y `E01`–`E10`.
5. **Proyecto:** [clase integradora](proyecto_integrador/notebooks/00_clase_teoria_y_practica_proyecto_rag.ipynb) y los cuatro laboratorios del proyecto.

La progresión de los ejercicios es consistente en las cuatro lectures:

| Rango | Modalidad | Cómo usarlo |
| --- | --- | --- |
| `E01`–`E04` | Resuelto y guiado | Leé, ejecutá y cambiá parámetros para comprobar tus hipótesis. |
| `E05`–`E06` | Para resolver | Completá las consignas antes de consultar una solución propia. |
| `E07` | Inicial | Construí una versión pequeña desde cero. |
| `E08` | Avanzado | Incorporá una restricción o técnica de producción. |
| `E09`–`E10` | Profundización | Conectá la técnica con evaluación, API o una solución trazable. |

## Estructura de carpetas

```text
AI_ENGINEERING_M2/
├── graficos/                         # Referencias visuales originales (.excalidraw)
├── presentaciones/                   # Presentaciones de las clases (.pdf)
├── notebooks/                        # 44 notebooks de L1 a L4
│   ├── AEM2L1_embeddings_fragmentacion/
│   ├── AEM2L2_vector_stores/
│   ├── AEM2L3_implementacion_rag/
│   ├── AEM2L4_alternativas_open_source/
│   ├── README.md                      # Orden y uso de todos los notebooks
│   └── validate_notebooks.py          # Verificación de estructura y sintaxis
├── python_puro/
│   └── AEM2_python_exercises/         # 20 ejercicios de terminal (5 por lecture)
│       ├── common.py                  # Utilidades compartidas
│       ├── requirements.txt
│       └── AEM2L*/data/               # Datos y generadores reproducibles
├── proyecto_integrador/               # FAQ RAG PeopleFlow
│   ├── data/                          # Documento fuente y golden cases
│   ├── notebooks/                     # Clase 00 + 4 laboratorios guiados
│   ├── src/                           # Código modular de producción
│   ├── tests/                         # Pruebas unitarias sin consumo de API
│   ├── outputs/                       # Salidas locales no versionadas
│   ├── requirements.txt
│   └── README.md
└── README.md                          # Este mapa general
```

No se guardan claves, índices generados ni resultados locales en Git. Consultá [`.gitignore`](.gitignore) para ver los artefactos excluidos.

## Contenido de cada lecture

### L1 — Embeddings y fragmentación de texto

Carpeta: [`notebooks/AEM2L1_embeddings_fragmentacion/`](notebooks/AEM2L1_embeddings_fragmentacion/)

Parte del problema de las búsquedas por palabras clave y presenta el espacio vectorial, similitud coseno y producto punto. Luego trabaja chunking, overlap, metadata, Top-K, trazabilidad y evaluación con golden cases.

Las visualizaciones incluyen espacio semántico 2D, comparación coseno/producto punto, bloques con solapamiento y rankings de recuperación. La práctica de terminal construye un recuperador semántico, primero con ejemplos deterministas y después con embeddings de API opcionales.

### L2 — Bases de datos vectoriales

Carpeta: [`notebooks/AEM2L2_vector_stores/`](notebooks/AEM2L2_vector_stores/)

Explica el registro vectorial, normalización y k-NN exacto. A continuación introduce FAISS y los compromisos de Flat, HNSW e IVF: exactitud, memoria, tiempo de construcción y latencia de consulta.

Las prácticas implementan ranking, persistencia y filtros por metadata; terminan comparando Recall@K y latencia sobre datos reproducibles.

### L3 — Implementación de RAG

Carpeta: [`notebooks/AEM2L3_implementacion_rag/`](notebooks/AEM2L3_implementacion_rag/)

Organiza el sistema en dos flujos: **ingesta** (limpiar, fragmentar, embeber e indexar) e **inferencia** (recuperar, construir contexto y responder). Trata grounding, Top-K, umbral, reranking, caché, fallas previsibles, latencia y evaluación.

La salida pública se estudia como contrato: una respuesta debe tener exactamente `user_question`, `system_answer` y `chunks_related`. Cuando no existe evidencia suficiente, el sistema debe decirlo explícitamente, no inventar una respuesta.

### L4 — Alternativas y ecosistema open-source

Carpeta: [`notebooks/AEM2L4_alternativas_open_source/`](notebooks/AEM2L4_alternativas_open_source/)

Convierte una decisión tecnológica en una decisión medible. Compara API y modelos locales según calidad, costo, privacidad, latencia, throughput y complejidad operativa. También aborda arquitectura híbrida, benchmark reproducible, shadow traffic y ADR.

El objetivo no es elegir una herramienta “mejor” en abstracto: es justificar una elección con datos y restricciones concretas.

## Notebooks: cómo trabajarlos

Los notebooks son material de estudio, no guiones para quien da la clase. Cada uno mantiene una estructura pedagógica común:

1. Problema real y objetivo.
2. Referencia visual y conceptual tomada de las presentaciones o diagramas del módulo.
3. Teoría explicada con una visual reproducible en Matplotlib o un diagrama en Mermaid.
4. Predicción: una pregunta para responder antes de ejecutar código.
5. Práctica incremental y resultado esperado.
6. Errores frecuentes, mini desafío y conexión con el siguiente tema.

Los gráficos matemáticos funcionan sin una clave de API. Las celdas que llaman a OpenAI están señaladas y se omiten con un mensaje claro si falta `OPENAI_API_KEY`.

Para abrirlos:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r python_puro\AEM2_python_exercises\requirements.txt
jupyter notebook
```

Desde Jupyter, abrí primero el notebook `00` de la lecture correspondiente.

## Ejercicios de Python puro

Carpeta principal: [`python_puro/AEM2_python_exercises/`](python_puro/AEM2_python_exercises/)

Son 20 scripts, cinco por lecture. Sirven para practicar fuera de Jupyter, leer entradas y salidas claramente y ejecutar los generadores de datos de cada bloque.

| Lecture | Qué se implementa en los scripts |
| --- | --- |
| L1 | Similitud, chunks con metadata, embeddings opcionales, golden cases y recuperador integrador. |
| L2 | k-NN exacto, FAISS Flat persistente, HNSW/IVF, filtros y benchmark. |
| L3 | Ingesta, respuestas con grounding/JSON, caché y reranking, evaluación y RAG integrador. |
| L4 | Matriz de decisión, latencia de API, estimación de costo, plan local/API y reporte. |

Configuración inicial:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r python_puro\AEM2_python_exercises\requirements.txt
Copy-Item python_puro\AEM2_python_exercises\.env.example python_puro\AEM2_python_exercises\.env
```

Ejemplo sin API:

```powershell
python python_puro\AEM2_python_exercises\AEM2L1_embeddings_fragmentacion\e01_similitud_manual.py
```

Cada carpeta de lecture tiene su propio `README.md`, datos de ejemplo y, cuando corresponde, un script `data/generate_data.py` para regenerar el corpus o las consultas.

## Proyecto integrador — PeopleFlow FAQ RAG

Carpeta: [`proyecto_integrador/`](proyecto_integrador/)

El proyecto lleva los conceptos a un asistente para preguntas frecuentes de una plataforma ficticia de RR.HH. Usa un documento local extenso, lo fragmenta de manera trazable y compara dos backends con los mismos embeddings:

| Componente | Rol |
| --- | --- |
| `data/faq_document.txt` | Fuente local de conocimiento. |
| `src/chunking.py` | Limpieza y fragmentación con tamaño y solapamiento configurables. |
| `src/embeddings.py` | Adaptador de embeddings; permite pruebas sin créditos mediante falsos. |
| `src/stores.py` | Interfaz `VectorStoreBackend`, Chroma persistente y FAISS + metadata JSON. |
| `src/index.py` | Generación, validación y carga de índices. |
| `src/query.py` | Retrieval, grounding, generación y contrato de respuesta. |
| `src/evaluate.py` / `benchmark.py` | Golden cases y comparación de backends. |
| `tests/` | Validaciones deterministas de chunking, contrato y salidas. |

El contrato de respuesta público es intencionalmente pequeño y exacto:

```json
{
  "user_question": "¿Cómo solicito vacaciones?",
  "system_answer": "...respuesta fundada únicamente en el contexto...",
  "chunks_related": ["chunk_...", "chunk_..."]
}
```

El `system_answer` se abstiene explícitamente si los chunks recuperados no ofrecen evidencia suficiente. Los ids de `chunks_related` permiten inspeccionar de dónde salió cada respuesta.

### Ejecutar el proyecto

```powershell
Set-Location proyecto_integrador
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Completar OPENAI_API_KEY en .env solo para embeddings y generación reales.

python -m src.generate_data
python -m src.index --backend all
python -m src.query --backend chroma --question "¿Cómo restablezco mi contraseña?"
python -m src.benchmark
python -m unittest discover -s tests -v
```

Los cuatro laboratorios de [`proyecto_integrador/notebooks/`](proyecto_integrador/notebooks/) muestran la arquitectura, indexación, retrieval/grounding y evaluación sin duplicar el código de producción.

## Configuración y uso de OpenAI

La ruta principal usa los siguientes valores, ambos configurables en `.env`:

| Variable | Valor por defecto | Uso |
| --- | --- | --- |
| `OPENAI_API_KEY` | Vacío | Habilita las llamadas reales a la API. |
| `AEM2_EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings de documentos y consultas. |
| `AEM2_GENERATION_MODEL` | `gpt-5.6-luna` | Generación fundamentada de la respuesta RAG. |
| `AEM2_CHUNK_SIZE` | `120` | Tamaño objetivo del chunk en el proyecto. |
| `AEM2_CHUNK_OVERLAP` | `24` | Solapamiento entre chunks. |
| `AEM2_TOP_K` | `4` | Cantidad de chunks recuperados por defecto. |

No compartas ni subas archivos `.env`. Trabajá sin clave mientras estudias los conceptos y activá la API solamente en las actividades señaladas. Las llamadas pueden generar costos según el uso y el precio vigente del proveedor.

## Validación y calidad

El módulo incluye verificaciones para que el material se mantenga ejecutable:

```powershell
# Desde la raíz del repositorio
python notebooks\validate_notebooks.py

# Desde proyecto_integrador
python -m unittest discover -s tests -v
```

La validación de notebooks comprueba su formato, secciones didácticas obligatorias y sintaxis Python de todas las celdas. Las pruebas del proyecto cubren documento vacío, rango de chunks, contrato público de tres claves y resultados de ejemplo. Las pruebas unitarias no requieren `OPENAI_API_KEY` ni consumen créditos.

## Material visual de referencia

Las explicaciones y las visualizaciones reproducibles de los notebooks se basan en las presentaciones y diagramas entregados con el módulo:

- [`presentaciones/`](presentaciones/) contiene los PDFs de L1, L2 y el proyecto integrador.
- [`graficos/`](graficos/) contiene los diagramas Excalidraw de L1–L4 y del proyecto.

Las imágenes originales no se incrustan en los notebooks: se recrean como gráficos, tablas y diagramas editables para que cada idea pueda explorarse ejecutando código.

## Criterios de finalización sugeridos

Considerá que completaste el módulo cuando puedas demostrar lo siguiente:

- Resolver los ejercicios `E05`–`E10` de las cuatro lectures.
- Explicar por qué un resultado aparece primero en un ranking vectorial.
- Medir al menos Recall@K y latencia para justificar un índice.
- Mostrar una respuesta RAG con sus chunks relacionados y una respuesta de abstención.
- Comparar Chroma y FAISS sobre el mismo corpus.
- Defender una decisión API/local/híbrida con una matriz ponderada y un ADR.

## Límites intencionales

Este módulo prioriza comprensión, trazabilidad y experimentación reproducible. Un sistema productivo real puede requerir autenticación, observabilidad, control de acceso por documento, protección de datos personales, monitoreo de calidad, evaluación humana y políticas específicas de seguridad. Esos temas se deben incorporar antes de usar un RAG con información sensible en producción.
