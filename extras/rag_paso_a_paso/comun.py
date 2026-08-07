"""Utilidades compartidas por los pasos: rutas, cliente OpenAI y normalizacion.

No es un "paso" en si mismo: es la caja de herramientas que los pasos siguientes
van a ir importando y reusando.
"""
from pathlib import Path
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.append(str(PROJECT_ROOT / "extras" / "embeddings" / "data"))

from documentos import DOCUMENTOS, PREGUNTA  # noqa: E402

ESTADO = Path(__file__).resolve().parent / "estado"
ESTADO.mkdir(exist_ok=True)

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

client = OpenAI()

# A que departamento pertenece cada fuente, para poder practicar filtros mas adelante.
DEPARTAMENTO_POR_FUENTE = {
    "manual_red.md": "IT",
    "facturacion.md": "Finance",
    "seguridad.md": "Security",
}


def normalizar(vectores: np.ndarray) -> np.ndarray:
    """Divide cada vector por su norma, para que el producto interno equivalga al coseno."""
    normas = np.linalg.norm(vectores, axis=1, keepdims=True)
    return vectores / normas


def embed(texto: str) -> np.ndarray:
    """Genera el embedding de un texto (se usa tanto para chunks como para preguntas)."""
    respuesta = client.embeddings.create(model=EMBEDDING_MODEL, input=texto)
    return np.array(respuesta.data[0].embedding)
