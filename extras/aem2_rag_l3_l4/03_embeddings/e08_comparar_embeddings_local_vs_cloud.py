# Este archivo no genera embeddings: prepara una comparacion para elegir proveedor. Revisa si
# estan disponibles OpenAI y Hugging Face, y recuerda que la eleccion se mide por calidad,
# latencia, costo y privacidad. Puede ejecutarse aun sin instalar los proveedores.
# os lee OPENAI_API_KEY desde las variables del entorno.
import os
# sys permite importar helpers compartidos desde un archivo independiente.
import sys
# Path encuentra la raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# optional_import comprueba si el proveedor local esta instalado.
from shared.utils import optional_import

# LangChain intercambia proveedores porque ambos respetan la interfaz Embeddings.
# La decision real debe considerar calidad de retrieval, latencia, costo y privacidad.
local_ready = optional_import("langchain_huggingface", "langchain-huggingface sentence-transformers")
cloud_ready = bool(os.getenv("OPENAI_API_KEY"))

print("Proveedor       API key  Descarga modelo  Estado")
print(f"OpenAI          si       no              {'listo' if cloud_ready else 'falta OPENAI_API_KEY'}")
print(f"HuggingFace     no       si              {'listo' if local_ready else 'falta instalacion'}")
print("Compara ambos con las mismas preguntas etiquetadas antes de elegir uno.")

# Resumen final: local y cloud se intercambian facilmente gracias a la interfaz Embeddings.
# La eleccion correcta se toma midiendo calidad, latencia, costo y requisitos de privacidad.
