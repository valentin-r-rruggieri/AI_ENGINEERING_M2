# Este archivo simula un circuit breaker sin depender de una API. Cuenta timeouts consecutivos,
# abre el circuito cuando llega al limite y cambia las siguientes llamadas a fallback local. Al
# ejecutarlo se ve cada transicion de estado y por que esta tecnica protege la latencia del RAG.
# El circuit breaker evita insistir con una dependencia que esta fallando repetidamente.
failures, limit, state = 0, 2, "CERRADO"

for attempt in range(1, 5):
    if state == "ABIERTO":
        print(f"Intento {attempt}: circuito abierto -> fallback local")
        continue

    # El timeout es simulado para que se vea como el contador cambia de estado.
    try:
        raise TimeoutError("timeout cloud simulado")
    except TimeoutError:
        failures += 1
        state = "ABIERTO" if failures >= limit else "CERRADO"
        print(f"Intento {attempt}: fallo {failures}/{limit}; estado={state}")

print("Abrir el circuito protege latencia y recursos hasta que la dependencia pueda recuperarse.")

# Resumen final: el circuit breaker deja de llamar temporalmente a una dependencia ya fallida.
# Asi protege recursos y permite responder rapido mediante una ruta alternativa o un mensaje claro.
