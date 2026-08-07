from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def embedding(texto):
    res = client.embeddings.create(
        model="nvidia/nemotron-3-embed-1b:free",
        input=texto,
        encoding_format="float",
    )

    return np.array(res.data[0].embedding)


def similitud_coseno(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


perro = embedding("perro")
gato = embedding("gato")
computadora = embedding("computadora")


print(
    "perro-gato:",
    similitud_coseno(perro, gato)
)

print(
    "perro-computadora:",
    similitud_coseno(perro, computadora)
)
