# Este archivo carga un modelo generativo de Transformers dentro de la maquina y lo adapta como
# LLM de LangChain con HuggingFacePipeline. Necesita transformers, torch y Hugging Face; la
# primera ejecucion descarga google/flan-t5-small. Al terminar responde una pregunta corta local.
# sys permite importar el helper que revisa dependencias opcionales.
import sys
# Path encuentra la raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# optional_import evita descargar o importar modelos si aun falta algun paquete.
from shared.utils import optional_import

hf_ready = optional_import("langchain_huggingface", "langchain-huggingface")
transformers_ready = optional_import("transformers", "transformers torch accelerate")

if hf_ready and transformers_ready:
    # pipeline crea el pipeline de generacion de Transformers que correra localmente.
    from transformers import pipeline
    # HuggingFacePipeline adapta ese pipeline al contrato de LLM de LangChain.
    from langchain_huggingface import HuggingFacePipeline

    model_name = "google/flan-t5-small"
    generator = pipeline("text2text-generation", model=model_name, max_new_tokens=80)
    llm = HuggingFacePipeline(pipeline=generator)

    # invoke es la misma forma de llamar a un LLM local o cloud dentro de LangChain.
    print(llm.invoke("Responde en una frase: que significa RAG?"))
    print("El primer uso descargara el modelo; despues Transformers lo reutiliza desde su cache local.")

# Resumen final: HuggingFacePipeline permite invocar un modelo Transformers como cualquier LLM de LangChain.
# Un modelo generativo local requiere mas memoria que embeddings, por eso conviene empezar con uno pequeno.
