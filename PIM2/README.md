# PeopleFlow FAQ RAG

PeopleFlow FAQ RAG es el proyecto integrador del módulo 2. Convierte documentación interna de una plataforma ficticia de Recursos Humanos en una base de conocimiento consultable. Primero fragmenta el FAQ, genera embeddings con OpenAI y los persiste en Chroma. Luego recupera evidencia semánticamente relevante antes de pedir una respuesta al modelo. Así, el asistente puede explicar qué chunks respaldan la respuesta y abstenerse cuando el documento no contiene información suficiente, sin reentrenar un modelo.

## Arquitectura

~~~
faq_document.txt → limpieza → chunks → embeddings OpenAI → Chroma
pregunta → embedding → retrieval Top-K → contexto → ChatOpenAI → JSON público
~~~

La indexación y la consulta son etapas separadas: el índice se puede reconstruir después de modificar el FAQ sin cambiar la lógica de preguntas.

## Requisitos e instalación

Requiere Python 3.10 o superior y una clave de OpenAI con acceso a embeddings y chat.

~~~powershell
cd PIM2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
~~~

Editá el archivo .env y completá OPENAI_API_KEY. Nunca subas ese archivo al repositorio.

## Ejecución

Primero construí el índice local:

~~~powershell
python src/build_index.py
~~~

Después hacé una consulta:

~~~powershell
python src/query.py --question "¿Cómo restablezco mi contraseña?"
~~~

El comando imprime un JSON con exactamente user_question, system_answer y chunks_related.

Para evaluar una respuesta ya guardada, pasá el archivo JSON con sus chunks:

~~~powershell
python src/query.py --question "¿Cómo restablezco mi contraseña?" > outputs/sample_response.json
python src/evaluator.py --question "¿Cómo restablezco mi contraseña?" --answer "Respuesta a evaluar" --chunks-file outputs/sample_response.json
~~~

El evaluador bonus devuelve independientemente score (0 a 10) y reason (mínimo 50 caracteres). Evalúa relevancia, grounding y completitud; no altera la salida pública del chatbot.

## Configuración y decisiones técnicas

| Variable | Valor predeterminado | Uso |
| --- | --- | --- |
| EMBEDDING_MODEL | text-embedding-3-small | Modelo usado tanto para corpus como para consultas. |
| CHAT_MODEL | gpt-4o-mini | Modelo que redacta la respuesta fundamentada. |
| CHUNK_SIZE | 300 | Tamaño de fragmento en caracteres, suficiente para conservar ideas cortas. |
| CHUNK_OVERLAP | 50 | Contexto compartido entre fragmentos contiguos. |
| TOP_K | 4 | Cantidad de chunks recuperados; debe estar entre 2 y 5. |
| CHROMA_COLLECTION | peopleflow_faq | Nombre de la colección persistida. |

El fragmentador recursivo prioriza encabezados y párrafos. Chroma implementa búsqueda vectorial por similitud y conserva la metadata source y chunk_id para trazabilidad. El prompt exige usar únicamente el contexto recuperado y declarar falta de evidencia cuando corresponda.

## Pruebas y ejemplos

~~~powershell
pytest
~~~

Las pruebas no consumen API: validan limpieza, chunking, parámetros, contratos JSON, ejemplos y evaluación. El archivo outputs/sample_queries.json incluye tres consultas de ejemplo: procedimiento, política e importación.

## Limitaciones

La calidad de las respuestas depende de la documentación y de la recuperación semántica. El sistema no reemplaza una revisión humana para políticas sensibles, y las llamadas a OpenAI requieren conectividad, credenciales válidas y pueden generar costos.
