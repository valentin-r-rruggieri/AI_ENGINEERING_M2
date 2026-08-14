# Este archivo explica nDCG con una lista pequena de relevancias graduadas. Sirve cuando un
# documento puede ser parcialmente util y cuando su posicion importa. Al ejecutarlo se calcula
# el ranking observado, el ideal y el valor normalizado que permite compararlos.
# math aporta log2, usado para descontar relevancia segun la posicion en el ranking.
import math


def dcg(relevances):
    # DCG descuenta posiciones tardias: una evidencia util vale mas al comienzo de la lista.
    return sum(relevance / math.log2(index + 2) for index, relevance in enumerate(relevances))


# nDCG compara el orden recuperado con el orden ideal de las mismas relevancias graduadas.
retrieved = [3, 0, 1, 2]
ideal = sorted(retrieved, reverse=True)
print("Relevancias recuperadas:", retrieved)
print(f"DCG={dcg(retrieved):.3f} IDCG={dcg(ideal):.3f} nDCG={dcg(retrieved)/dcg(ideal):.3f}")
print("nDCG sirve cuando la relevancia no es solo si/no, sino que tiene distintos niveles.")

# Resumen final: nDCG valora tanto la utilidad de un resultado como su posicion en el ranking.
# Es apropiada cuando hay grados de relevancia y no solo una etiqueta binaria de correcto o no.
