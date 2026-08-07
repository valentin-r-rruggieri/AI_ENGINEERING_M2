# Ejercicios simples de embeddings y chunking

Carpetas:

- `manual/`: todo escrito a mano con OpenAI SDK + NumPy.
- `langchain/`: el mismo flujo usando LangChain.
- `llamaindex/`: el mismo flujo usando LlamaIndex.

Cada carpeta tiene:

- un notebook paso a paso;
- un archivo `.py` para correr todo junto.

Dataset: `data/documentos.py`, tomado de la actividad practica de la lecture.

## Instalar

```bash
& ".\.venv\Scripts\python.exe" -m pip install -r ".\embeddings\requirements.txt"
```

La key se toma del `.env` que ya existe en la raiz del proyecto.

## Ejecutar los scripts

```bash
& ".\.venv\Scripts\python.exe" ".\embeddings\manual\ejemplo_manual.py"
& ".\.venv\Scripts\python.exe" ".\embeddings\langchain\ejemplo_langchain.py"
& ".\.venv\Scripts\python.exe" ".\embeddings\llamaindex\ejemplo_llamaindex.py"
```
