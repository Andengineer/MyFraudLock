# api/ml_utils.py
import json, math, datetime as dt
from .ml.xai import predict_and_explain

def _derive_time_parts(fecha: dt.datetime):
    if not fecha:
        fecha = dt.datetime.now()
    hour = fecha.hour
    weekday = fecha.weekday()          # 0=Lunes..6=Domingo
    month = fecha.month
    is_weekend = 1 if weekday >= 5 else 0
    return hour, weekday, month, is_weekend

def predict_fraud(tx):
    """
    tx: instancia Transaccion o dict con:
      importe, fecha, category, state, gender, age, city_pop
    """
    get = (lambda k: getattr(tx, k)) if not isinstance(tx, dict) else (lambda k: tx.get(k))

    amt = float(get("importe") or 0.0)
    amt_log1p = math.log1p(max(0.0, amt))
    hour, weekday, month, is_weekend = _derive_time_parts(get("fecha"))
    age = int(get("age") or 0)
    city_pop = int(get("city_pop") or 0)
    category = (get("category") or "").strip()
    state    = (get("state") or "").strip()
    gender   = (get("gender") or "").strip().lower()

    payload = {
        "amt": amt,
        "amt_log1p": amt_log1p,
        "age": age,
        "hour": hour,
        "weekday": weekday,
        "month": month,
        "is_weekend": is_weekend,
        "city_pop": city_pop,
        "category": category,
        "state": state,
        "gender": gender,
    }
    score, exp = predict_and_explain(payload)
    return score, json.dumps(exp, ensure_ascii=False)
