# api/ml_utils.py
"""
Utilidades ML para detección de fraude.
Conecta el modelo Transacción de Django con el modelo DL entrenado.
"""
import json, math, datetime as dt
import numpy as np
from .ml.xai import predict_and_explain
from django.utils import timezone


def _derive_time_parts(fecha):
    """Deriva features temporales usando tz local."""
    if not fecha:
        fecha = timezone.now()
    local = timezone.localtime(fecha)
    hour = local.hour
    weekday = local.weekday()          # 0=Lun..6=Dom
    month = local.month
    is_weekend = 1 if weekday >= 5 else 0
    return hour, weekday, month, is_weekend


def predict_fraud(tx):
    """
    tx: instancia Transaccion o dict con las features del nuevo modelo.

    Features requeridas:
      importe, fecha, card_brand, card_type, issuer_bank, payment_channel,
      eci_code, num_installments, customer_region, city_population,
      is_new_customer, days_since_first_purchase, avg_historical_amount,
      category, num_items, has_discount, previous_failed_attempts
    """
    get = (lambda k: getattr(tx, k)) if not isinstance(tx, dict) else (lambda k: tx.get(k))

    # Monto
    amt = float(get("importe") or 0.0)
    amt_log1p = math.log1p(max(0.0, amt))

    # Temporales
    hour, weekday, month, is_weekend = _derive_time_parts(get("fecha"))
    is_high_risk_hour = 1 if hour in (0, 1, 2, 3, 4, 5) else 0

    # Ciclicidad temporal
    hour_sin = round(math.sin(2 * math.pi * hour / 24), 4)
    hour_cos = round(math.cos(2 * math.pi * hour / 24), 4)
    day_sin  = round(math.sin(2 * math.pi * weekday / 7), 4)
    day_cos  = round(math.cos(2 * math.pi * weekday / 7), 4)
    month_sin = round(math.sin(2 * math.pi * month / 12), 4)
    month_cos = round(math.cos(2 * math.pi * month / 12), 4)

    # Tarjeta y pago
    card_brand      = (get("card_brand") or "visa").strip().lower()
    card_type       = (get("card_type") or "debit").strip().lower()
    issuer_bank     = (get("issuer_bank") or "bcp").strip().lower()
    payment_channel = (get("payment_channel") or "web").strip().lower()
    eci_code        = int(get("eci_code") or 5)
    has_3ds         = 1 if eci_code in (5, 2) else 0
    num_installments = int(get("num_installments") or 0)

    # Cliente
    customer_region = (get("customer_region") or "lima").strip().lower()
    city_population = int(get("city_population") or 0)
    is_new_customer = 1 if get("is_new_customer") else 0
    days_since_first = int(get("days_since_first_purchase") or 0)
    avg_hist_amt    = float(get("avg_historical_amount") or 0.0)

    # Pedido
    category  = (get("category") or "otros").strip().lower()
    num_items = int(get("num_items") or 1)
    has_discount = 1 if get("has_discount") else 0
    prev_failed = int(get("previous_failed_attempts") or 0)

    # Desviación de monto
    amount_deviation = round((amt - avg_hist_amt) / (avg_hist_amt + 1e-6), 4)
    amount_deviation = max(-10.0, min(50.0, amount_deviation))

    # ─── Nuevas features derivadas ───
    amt_hour_interaction = round(amt_log1p * hour_sin, 4)
    amt_fail_interaction = round(math.tanh(amt / 500) * math.log1p(prev_failed), 4)
    risk_score_smooth = round(math.tanh(
        amount_deviation * 0.3 +
        prev_failed * 0.5 +
        is_new_customer * 0.4 +
        (1 - has_3ds) * 0.3 +
        is_high_risk_hour * 0.2
    ), 4)
    amt_pop_ratio = amt / (city_population + 1)
    amt_pop_sigmoid = round(1 / (1 + math.exp(-10 * (amt_pop_ratio - 0.001))), 4)
    customer_maturity = round(math.tanh(days_since_first / 365), 4)
    night_newcust_score = round(
        (1 - customer_maturity) * (1 - math.cos(2 * math.pi * hour / 24)) / 2, 4
    )

    # ─── Biometría Conductual y Telemetría ───
    # Si no vienen en el request (ej. simulación manual antigua), simulamos valores legítimos promedio
    t = np.random.uniform(-0.5, 0.5)
    base_duration = t + np.random.normal(0, 0.25)
    base_velocity = t + np.random.normal(0, 0.25)

    session_duration = float(get("session_duration_minutes") or round(max(0.5, min(60.0, base_duration * 15 + 15)), 2))
    interaction_vel = float(get("interaction_velocity") or round(max(0.5, min(100.0, base_velocity * 25 + 25)), 2))

    device_tel_1 = float(get("device_telemetry_1") or round(np.random.normal(-1.0, 1.0), 4))
    device_tel_2 = float(get("device_telemetry_2") or round(np.random.normal(0, 2.0), 4))
    device_tel_3 = float(get("device_telemetry_3") or round(np.random.normal(0, 2.0), 4))
    device_tel_4 = float(get("device_telemetry_4") or round(np.random.normal(0, 2.0), 4))
    device_tel_5 = float(get("device_telemetry_5") or round(np.random.normal(0, 2.0), 4))

    payload = {
        "transaction_amount": amt,
        "amt_log1p":          amt_log1p,
        "hour":               hour,
        "day_of_week":        weekday,
        "month":              month,
        "is_weekend":         is_weekend,
        "eci_code":           eci_code,
        "has_3ds":            has_3ds,
        "city_population":    city_population,
        "num_items":          num_items,
        "has_discount":       has_discount,
        "num_installments":   num_installments,
        "previous_failed_attempts": prev_failed,
        "is_new_customer":    is_new_customer,
        "days_since_first_purchase": days_since_first,
        "avg_historical_amount": avg_hist_amt,
        "is_high_risk_hour":  is_high_risk_hour,
        "amount_deviation":   amount_deviation,
        "hour_sin":           hour_sin,
        "hour_cos":           hour_cos,
        "day_sin":            day_sin,
        "day_cos":            day_cos,
        "month_sin":          month_sin,
        "month_cos":          month_cos,
        "card_brand":         card_brand,
        "card_type":          card_type,
        "issuer_bank":        issuer_bank,
        "payment_channel":    payment_channel,
        "customer_region":    customer_region,
        "category":           category,
        "amt_hour_interaction": amt_hour_interaction,
        "amt_fail_interaction": amt_fail_interaction,
        "risk_score_smooth":  risk_score_smooth,
        "amt_pop_sigmoid":    amt_pop_sigmoid,
        "customer_maturity":  customer_maturity,
        "night_newcust_score": night_newcust_score,
        "session_duration_minutes": session_duration,
        "interaction_velocity": interaction_vel,
        "device_telemetry_1": device_tel_1,
        "device_telemetry_2": device_tel_2,
        "device_telemetry_3": device_tel_3,
        "device_telemetry_4": device_tel_4,
        "device_telemetry_5": device_tel_5,
    }

    score, exp = predict_and_explain(payload)
    return score, json.dumps(exp, ensure_ascii=False)
