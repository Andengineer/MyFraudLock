import urllib.request
import json
import random
import time

URL = 'http://127.0.0.1:8000/api/transacciones/'

def post_transaction(data, label):
    req = urllib.request.Request(
        URL, 
        data=json.dumps(data).encode('utf8'), 
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            print(f"[{label}] OK - Transacción #{res_data.get('id_transaccion')} guardada.")
    except urllib.error.HTTPError as e:
        print(f"[{label}] Error HTTP {e.code}:", e.read().decode('utf-8'))
    except Exception as e:
        print(f"[{label}] Error:", e)

if __name__ == '__main__':
    print("Enviando transacciones de prueba al API...")
    
    # Transacciones Normales
    for i in range(3):
        normal_tx = {
            'importe': round(random.uniform(50.0, 300.0), 2),
            'card_brand': random.choice(['visa', 'mastercard']),
            'card_type': 'debit',
            'issuer_bank': 'bcp',
            'payment_channel': 'web',
            'eci_code': 5,
            'num_installments': 0,
            'customer_region': 'lima',
            'city_population': 10000000,
            'is_new_customer': False,
            'days_since_first_purchase': random.randint(100, 300),
            'avg_historical_amount': round(random.uniform(100.0, 500.0), 2),
            'category': 'otros',
            'num_items': 1,
            'has_discount': False,
            'previous_failed_attempts': 0,
            # Simulando biometría normal: sesión larga, velocidad baja
            'session_duration_minutes': round(random.uniform(5.0, 15.0), 2),
            'interaction_velocity': round(random.uniform(10.0, 30.0), 2)
        }
        post_transaction(normal_tx, f"Normal {i+1}")
        time.sleep(0.5)

    # Transacciones Fraudulentas
    for i in range(2):
        fraud_tx = {
            'importe': round(random.uniform(4000.0, 8000.0), 2), # Monto anómalo alto
            'card_brand': 'other',
            'card_type': 'credit',
            'issuer_bank': 'otros',
            'payment_channel': 'mobile',
            'eci_code': 7, # Sin 3DS
            'num_installments': 1,
            'customer_region': 'piura',
            'city_population': 500000,
            'is_new_customer': True, # Nuevo
            'days_since_first_purchase': 0,
            'avg_historical_amount': 50.0, # Historial muy bajo
            'category': 'electronica',
            'num_items': random.randint(3, 8),
            'has_discount': False,
            'previous_failed_attempts': random.randint(3, 6), # Intentos previos
            # Biometría fraudulenta: sesión cortísima, alta velocidad
            'session_duration_minutes': round(random.uniform(0.1, 0.8), 2),
            'interaction_velocity': round(random.uniform(85.0, 120.0), 2),
            # Telemetría fraudulenta
            'device_telemetry_1': round(random.uniform(-5.0, -3.0), 2),
            'device_telemetry_2': round(random.uniform(4.0, 6.0), 2)
        }
        post_transaction(fraud_tx, f"Fraude {i+1}")
        time.sleep(0.5)
        
    print("Prueba finalizada.")
