# PeopleFlow FAQ — Proyecto integrador M2

Asistente RAG para consultas internas de una plataforma ficticia de RR.HH. El proyecto enseña todo el flujo: documento → chunks → embeddings → dos vector stores → contexto → respuesta fundamentada.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# completar OPENAI_API_KEY en .env
```

## Uso

```powershell
python -m src.generate_data
python -m src.index --backend all
python -m src.query --backend chroma --question "¿Cómo restablezco mi contraseña?"
python -m src.benchmark
python -m unittest discover -s tests -v
```

## Decisiones

- **Chunking:** 120 palabras con overlap 24; el corpus queda en chunks trazables de 50 a 500 tokens aproximados.
- **Embeddings:** `text-embedding-3-small`, igual para corpus y consulta.
- **Vector stores:** Chroma persistente y FAISS + metadata JSON; reciben el mismo vector para una comparación justa.
- **Retrieval:** coseno / producto interno sobre vectores normalizados, Top-K configurable (4).
- **Generación:** `gpt-5.6-luna` por API Responses, con grounding estricto.
- **Contrato:** la salida pública contiene exactamente `user_question`, `system_answer` y `chunks_related`.

## Laboratorios guiados

La carpeta `notebooks/` acompaña la implementación sin duplicar su lógica:

1. Arquitectura y contrato JSON.
2. Documento, chunking, embeddings e indexación.
3. Retrieval, grounding y respuesta trazable.
4. Evaluación, benchmark y preparación de defensa.

Ejecutá los laboratorios desde `proyecto_integrador/` después de instalar las dependencias.

## Límites

La evaluación semántica de la respuesta depende de un modelo y puede variar; las pruebas automatizadas verifican de forma determinista chunking, persistencia, ranking, validaciones y contrato. Los índices y la clave no se versionan.
