# api/ml/xai.py
"""
Módulo de Explicabilidad (XAI) para el modelo de detección de fraude.
Usa SHAP (DeepExplainer / KernelExplainer) para generar explicaciones
de las predicciones del modelo Deep Learning.
"""
from pathlib import Path
import json, numpy as np, pandas as pd
from joblib import load
import shap
import tensorflow as tf

# --- SHIM de compatibilidad scikit-learn 1.6.x -> 1.7.x ---
# Evita: AttributeError: Can't get attribute '_RemainderColsList' ...
try:
    import sklearn.compose._column_transformer as _ctmod  # type: ignore
    if not hasattr(_ctmod, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ctmod._RemainderColsList = _RemainderColsList  # monkey-patch
except Exception:
    pass

ML_DIR = Path(__file__).resolve().parent
MODEL_PATH   = ML_DIR / "best_model.keras"
PREPROC_PATH = ML_DIR / "preprocessor.joblib"
FEATS_PATH   = ML_DIR / "feature_names.json"
BG_PATH      = ML_DIR / "background.npy"
GROUPMAP_PATH = ML_DIR / "group_map.json"

# Carga lazy (en la primera llamada)
_model = None
_preproc = None
_feature_names = None
_background = None
_group_map = None
_explainer = None


def _ensure_artifacts():
    """Carga artefactos y el explainer una sola vez (lazy)."""
    global _model, _preproc, _feature_names, _background, _group_map, _explainer
    if _model is not None and _preproc is not None and _feature_names is not None and _background is not None:
        if _explainer is None:
            _init_explainer()
        return

    # 1) Cargar artefactos
    _model = tf.keras.models.load_model(MODEL_PATH)
    _preproc = load(PREPROC_PATH)
    with open(FEATS_PATH, encoding="utf-8") as f:
        _feature_names = json.load(f)
    _background = np.load(BG_PATH)
    if GROUPMAP_PATH.exists():
        with open(GROUPMAP_PATH, encoding="utf-8") as f:
            _group_map = json.load(f)

    # 2) Inicializar explainer
    _init_explainer()


def _init_explainer():
    """DeepExplainer si se puede, KernelExplainer de fallback."""
    global _explainer
    if _explainer is not None:
        return
    try:
        _explainer = shap.DeepExplainer(_model, _background)
    except Exception:
        _explainer = shap.KernelExplainer(
            lambda X: _model.predict(X, verbose=0), _background
        )


def _transform(payload: dict) -> np.ndarray:
    """Transforma el payload crudo usando el preprocesador guardado."""
    df = pd.DataFrame([payload])
    X = _preproc.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X


# ── Mapeo de nombres legibles para explicabilidad ────────────────────
FEATURE_DISPLAY_NAMES = {
    "transaction_amount": "Monto Transacción",
    "amt_log1p": "Monto (log)",
    "amount_deviation": "Desviación de monto",
    "hour": "Hora",
    "hour_sin": "Hora (sen)",
    "hour_cos": "Hora (cos)",
    "day_of_week": "Día",
    "day_sin": "Día (sen)",
    "day_cos": "Día (cos)",
    "month": "Mes",
    "month_sin": "Mes (sen)",
    "month_cos": "Mes (cos)",
    "is_weekend": "Fin de Semana",
    "is_new_customer": "Cliente Nuevo",
    "is_high_risk_hour": "Hora de Riesgo",
    "eci_code": "Código ECI",
    "has_3ds": "Aprobación 3DS",
    "has_discount": "Descuento Aplicado",
    "city_population": "Población",
    "num_items": "Cantidad de Ítems",
    "num_installments": "Cuotas",
    "previous_failed_attempts": "Intentos Fallidos",
    "days_since_first_purchase": "Antigüedad (días)",
    "avg_historical_amount": "Prom. Histórico",
    "card_brand": "Marca de Tarjeta",
    "card_type": "Tipo de Tarjeta",
    "issuer_bank": "Banco Emisor",
    "payment_channel": "Canal de Pago",
    "customer_region": "Región",
    "category": "Categoría de Producto",
    "amt_hour_interaction": "Anomalía Monto/Hora",
    "amt_fail_interaction": "Anomalía Monto/Intentos Fallidos",
    "risk_score_smooth": "Puntaje de Riesgo Compuesto",
    "amt_pop_sigmoid": "Monto vs Población",
    "customer_maturity": "Madurez del Cliente",
    "night_newcust_score": "Riesgo Nocturno Nuevo Cliente",
    "session_duration_minutes": "Duración de Sesión (min)",
    "interaction_velocity": "Velocidad de Interacción (Biometría)",
    "device_telemetry_1": "Telemetría Dispositivo 1",
    "device_telemetry_2": "Telemetría Dispositivo 2",
    "device_telemetry_3": "Telemetría Dispositivo 3",
    "device_telemetry_4": "Telemetría Dispositivo 4",
    "device_telemetry_5": "Telemetría Dispositivo 5"
}


def _aggregate(feature_contrib: dict) -> dict:
    """
    Agrupa impactos One-Hot por feature base.
    """
    categories = [
        "card_brand", "card_type", "issuer_bank",
        "payment_channel", "customer_region", "category"
    ]
    agg = {}
    for name, val in feature_contrib.items():
        base = name
        for cat in categories:
            if name.startswith(cat + "_"):
                base = cat
                break
        agg[base] = agg.get(base, 0.0) + val
    return agg


def predict_and_explain(payload: dict, top_k: int = 6, aggregate: bool = True):
    """
    Recibe un dict con las features crudas y retorna (score, explanation).

    payload debe contener claves numéricas:
      transaction_amount, amt_log1p, hour, day_of_week, month, is_weekend,
      eci_code, has_3ds, city_population, num_items, has_discount,
      num_installments, previous_failed_attempts, is_new_customer,
      days_since_first_purchase, avg_historical_amount, is_high_risk_hour,
      amount_deviation, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos

    Y categóricas:
      card_brand, card_type, issuer_bank, payment_channel,
      customer_region, category
    """
    _ensure_artifacts()

    X = _transform(payload)
    prob = float(_model.predict(X, verbose=0).ravel()[0])
    score = float(prob * 100)

    # SHAP values
    shap_vals = _explainer.shap_values(X)
    sv = shap_vals[0][0] if isinstance(shap_vals, list) else shap_vals[0]
    sv = np.squeeze(sv)

    contrib = {name: float(sv[i]) for i, name in enumerate(_feature_names)}

    if aggregate:
        contrib = _aggregate(contrib)

    ordered = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]

    # Añadir nombres legibles y lógica de negocio
    factors = []
    explicacion_negocio = []
    
    for k, v in ordered:
        display = FEATURE_DISPLAY_NAMES.get(k, k.replace("_", " ").title())
        val_raw = payload.get(k, "—")
        factors.append({"feature": k, "display_name": display, "impact": v, "value": val_raw})

        # Generar reglas de negocio para los factores agravantes (aumentan riesgo)
        if v > 0.05:  # umbral mínimo para destacarlo narrativamente
            if k == "amount_deviation" and payload.get("amount_deviation", 0) > 0.2:
                explicacion_negocio.append("El monto supera significativamente el promedio histórico de compras de este usuario.")
            elif k == "is_new_customer" and payload.get("is_new_customer") == 1:
                explicacion_negocio.append("Es un cliente nuevo y sin historial verificado en la plataforma.")
            elif k == "previous_failed_attempts" and payload.get("previous_failed_attempts", 0) > 0:
                explicacion_negocio.append(f"Se detectaron {payload.get('previous_failed_attempts')} intentos de pago fallidos previos en corto tiempo.")
            elif k == "transaction_amount":
                # Only warn if the impact is significantly high
                explicacion_negocio.append(f"El monto transaccional neto representa un volumen inusual o de alto riesgo.")
            elif k == "is_high_risk_hour" and payload.get("is_high_risk_hour") == 1:
                explicacion_negocio.append("La transacción se intentó en un horario atípico considerado de alto riesgo (ej. madrugada).")
            elif k == "category":
                cat = str(payload.get("category", "")).replace("_", " ").title()
                explicacion_negocio.append(f"La categoría de producto ({cat}) refleja alta vulnerabilidad a intentos de fraude en este contexto.")
            elif k == "card_brand" or k == "issuer_bank":
                explicacion_negocio.append("El bin de tarjeta (emisor/franquicia) está correlacionado con esquemas de fraude detectados recientemente.")
            elif k == "has_3ds" and payload.get("has_3ds") == 0:
                explicacion_negocio.append("El pago carece de mecanismo de verificación dinámico 3D Secure (baja fricción).")
            elif k == "num_items" and payload.get("num_items", 0) > 2:
                explicacion_negocio.append("La cantidad múltiple de artículos sugiere comportamiento tipo 'acaparamiento' fraudulento.")
            elif k == "customer_region":
                explicacion_negocio.append("La ubicación o región del incidente cruza con zonas de concurrencia de fraude o desconectadas de sus patrones habituales.")
            elif k == "interaction_velocity":
                explicacion_negocio.append("La biometría conductual detectó una velocidad de interacción anómala (potencial uso de bots o scripts automatizados).")
            elif k == "night_newcust_score":
                explicacion_negocio.append("Se detectó un patrón de alto riesgo: cliente sin historial operando en horarios de madrugada.")
            elif k.startswith("device_telemetry_"):
                explicacion_negocio.append("Se detectaron anomalías en la telemetría multidimensional del dispositivo (firma del equipo no concuerda con usuario legítimo).")

    # Deduplicar
    explicacion_negocio = list(dict.fromkeys(explicacion_negocio))

    explanation = {
        "top_factors": factors,
        "explicacion_negocio": explicacion_negocio,
        "sum_abs": float(sum(abs(v) for v in contrib.values())),
        "prob": round(prob, 6),
    }
    return score, explanation
