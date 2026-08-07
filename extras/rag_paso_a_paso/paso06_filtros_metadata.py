"""Paso 6: filtrar los resultados de FAISS por metadata (por ejemplo, por departamento).

La similitud semantica no sabe de permisos: primero buscamos por parecido,
despues aplicamos la regla de negocio (que departamento puede ver que chunk).
"""
from comun import PREGUNTA
from paso05_indice_faiss import buscar_faiss


def buscar_con_filtro(pregunta: str, department: str, top_k: int = 3, candidatos: int = 10) -> list[dict]:
    """Recupera mas candidatos de los necesarios y se queda con los del departamento pedido."""
    candidatos_faiss = buscar_faiss(pregunta, top_k=candidatos)
    filtrados = [c for c in candidatos_faiss if c["metadata"]["department"] == department]
    return filtrados[:top_k]


if __name__ == "__main__":
    for department in ["IT", "Finance", "Security"]:
        print(f"\nPregunta: {PREGUNTA}  (usuario de {department})")
        resultados = buscar_con_filtro(PREGUNTA, department)
        if not resultados:
            print("  (sin resultados para este departamento)")
        for resultado in resultados:
            print(f"  [{resultado['score']:.4f}] ({resultado['metadata']['source']}) {resultado['text']}")
