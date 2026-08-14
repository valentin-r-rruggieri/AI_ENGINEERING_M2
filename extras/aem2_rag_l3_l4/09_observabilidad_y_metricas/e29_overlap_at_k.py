# Este archivo compara dos conjuntos de resultados para medir cuanto cambia un retriever despues
# de una migracion. Calcula overlap como interseccion sobre union y no requiere ningun modelo.
# Al ejecutarlo se ve cuando una nueva configuracion entrega evidencia muy distinta a la anterior.
# Overlap@K compara los IDs recuperados por dos configuraciones antes de migrar un modelo o indice.
model_a = {"vacaciones", "remoto", "horario"}
model_b = {"vacaciones", "horario", "salud"}

# La interseccion sobre la union deja ver cuanto cambia el conjunto de evidencia devuelto.
overlap = len(model_a & model_b) / len(model_a | model_b)
print("A:", sorted(model_a), "\nB:", sorted(model_b))
print(f"Overlap@3 (Jaccard): {overlap:.2f}")
print("Un overlap bajo no es automaticamente malo, pero si pide revisar calidad y regresiones.")

# Resumen final: overlap detecta cuanto cambia la evidencia al reemplazar modelos o indices.
# Debe combinarse con metricas de calidad porque una diferencia puede ser mejora o regresion.
