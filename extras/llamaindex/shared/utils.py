# Este modulo contiene helpers para mantener los ejercicios de LlamaIndex cortos y ejecutables.
# Los mocks permiten aprender la estructura del RAG antes de configurar proveedores reales.

# importlib.util detecta integraciones opcionales sin producir errores de import.
import importlib.util

# Settings configura proveedores globales y VectorStoreIndex construye el indice de LlamaIndex.
from llama_index.core import Settings, VectorStoreIndex
# MockEmbedding crea vectores locales para indexar sin API externa.
from llama_index.core.embeddings import MockEmbedding
# MockLLM responde de forma didactica para probar QueryEngine sin descargar un modelo.
from llama_index.core.llms.mock import MockLLM


def setup_mock() -> None:
    Settings.embed_model = MockEmbedding(embed_dim=384)
    Settings.llm = MockLLM(max_tokens=128)


def build_mock_index(documents):
    setup_mock()
    return VectorStoreIndex.from_documents(documents)


def show_nodes(items) -> None:
    for number, item in enumerate(items, start=1):
        node = getattr(item, "node", item)
        print(f"[{number}] score={getattr(item, 'score', None)} metadata={node.metadata}")
        print(node.get_content()[:220])


def optional_import(module: str, package: str) -> bool:
    if importlib.util.find_spec(module) is not None:
        return True
    print(f"Falta '{package}'. Instala con: pip install {package}")
    return False


# Resumen final: los helpers muestran LlamaIndex sin esconder su flujo de indexacion y retrieval.
