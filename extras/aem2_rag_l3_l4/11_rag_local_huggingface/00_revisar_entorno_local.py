# Este archivo es el punto de partida del recorrido local. Revisa si el equipo tiene las librerias
# necesarias para embeddings, FAISS y Transformers, sin descargar ningun modelo. Ejecutalo primero
# para saber que instalar; la salida marca LISTO o FALTA y muestra el comando de instalacion.
# importlib.util permite comprobar paquetes sin importarlos ni descargar modelos.
import importlib.util

# Este mapa deja claro que pieza hace falta para cada etapa local del RAG.
packages = {
    "langchain_huggingface": "conecta Hugging Face con LangChain",
    "sentence_transformers": "calcula embeddings locales",
    "faiss": "crea el indice vectorial local",
    "transformers": "carga el modelo generativo local",
    "torch": "ejecuta los tensores del modelo",
}

for package, purpose in packages.items():
    installed = importlib.util.find_spec(package) is not None
    print(f"{package:<25} {'LISTO' if installed else 'FALTA'} - {purpose}")

print("\nInstalacion sugerida: pip install langchain-huggingface sentence-transformers faiss-cpu transformers torch accelerate")
print("Un RAG local evita una API key, pero descarga modelos y consume memoria del equipo.")

# Resumen final: el entorno local necesita paquetes distintos para vectorizar, indexar y generar.
# Verificar todo antes evita confundir un error de instalacion con un problema del pipeline RAG.
