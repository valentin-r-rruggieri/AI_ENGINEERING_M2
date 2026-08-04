# AEM2 — AI Engineering Module 2: RAG

Material didáctico completo sobre embeddings, búsqueda vectorial y Retrieval-Augmented Generation (RAG).

## Estructura

- `notebooks/`: 40 notebooks autocontenidos, 10 por lecture.
- `python_puro/AEM2_python_exercises/`: 20 ejercicios ejecutables desde terminal.
- `proyecto_integrador/`: asistente FAQ de RR.HH. con Chroma y FAISS comparables.

## Lectures

1. **AEM2L1** — Embeddings y fragmentación de texto.
2. **AEM2L2** — Bases de datos vectoriales.
3. **AEM2L3** — Implementación de RAG.
4. **AEM2L4** — Alternativas y ecosistema open-source.

## Requisitos

Python 3.10+ y una variable `OPENAI_API_KEY` para los ejercicios que llaman a la API. Los ejemplos puramente algorítmicos y sus pruebas funcionan sin clave.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r python_puro\AEM2_python_exercises\requirements.txt
Copy-Item python_puro\AEM2_python_exercises\.env.example python_puro\AEM2_python_exercises\.env
```

La configuración predeterminada usa `text-embedding-3-small` y `gpt-5.6-luna`. Nunca subas un archivo `.env` con una clave real.
