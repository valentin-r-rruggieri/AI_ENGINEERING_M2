# Este archivo ensena que recuperar documentos no significa enviarlos todos al modelo. Construye
# un contexto con un presupuesto aproximado de palabras y deja afuera resultados que exceden
# ese limite. Al ejecutarlo se ve que evidencia entra y cuanto contexto se consume.
# sys permite resolver imports del paquete shared al ejecutar el archivo solo.
import sys
# Path localiza la carpeta raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS es la evidencia sobre la que se hara retrieval.
from shared.dataset import DOCUMENTS
# lexical_search recupera candidatos, tokens estima su costo y show_documents los imprime.
from shared.utils import lexical_search, show_documents, tokens

# Primero se recuperan candidatos. El contexto final no tiene por que incluirlos a todos.
results = lexical_search("vacaciones trabajo remoto horario", DOCUMENTS, k=4)

# Las palabras son una aproximacion simple a tokens para visualizar un presupuesto de contexto.
budget, used, selected = 22, 0, []
for document in results:
    cost = len(tokens(document.page_content))
    if used + cost <= budget:
        selected.append(document)
        used += cost

show_documents(selected)
print(f"\nPresupuesto usado: {used}/{budget} palabras aproximadas")
print("El limite de contexto obliga a priorizar evidencia antes de llamar al modelo.")

# Resumen final: el contexto del LLM es limitado y debe llenarse con evidencia priorizada.
# Controlar el presupuesto reduce costo y evita que documentos debiles desplacen a los utiles.
