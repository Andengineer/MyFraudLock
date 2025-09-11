import random

def predict_fraud(data):
    """
    Simula el comportamiento de un modelo de ML/DL para fraude.
    Devuelve score de riesgo y explicabilidad (features con peso).
    """
    # Score de riesgo aleatorio 0-100 (simulación)
    score = round(random.uniform(0, 100), 2)

    # Explicabilidad falsa (ejemplo con features que recibimos del request)
    explicabilidad = {
        "importe": round(random.uniform(0, 1), 2),
        "metodo_pago": round(random.uniform(0, 1), 2),
        "direccion_envio": round(random.uniform(0, 1), 2),
    }

    return score, explicabilidad
