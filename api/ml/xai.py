# api/ml/xai.py
"""
Inferencia + Explicabilidad (XAI) del modelo de detección de fraude.

Modelo: FraudDNN (PyTorch) — el modelo del paper.
  - Pesos:        fraud_dnn_weights.pt
  - Preprocesador: fraud_dnn_preprocessor.joblib (RobustScaler + OneHot + OrdinalEncoder)
  - Specs:        fraud_dnn_hparams.json, fraud_dnn_embedding_spec.json,
                  fraud_dnn_meta.json, fraud_dnn_features_final.json
  - Background:   fraud_dnn_background.npy (espacio procesado, para SHAP)

Esquema de entrada del modelo (26 columnas crudas):
  numéricas:  eci, codigo_de_accion, importe_pedido, numero_de_cuotas, cantidad_items,
              tiene_cupon, billing_mismatch, weekend_night_flag, phone_format_valid,
              tx_hour, C1_AAR..C7_DPE
  one-hot:    marca, tipo_de_tarjeta, canal_pago, metodo_envio, pago_con_yape, pago_con_plin
  embeddings: emisor, region_cliente, categoria_producto

Este módulo recibe el payload de la aplicación (claves en inglés que produce
ml_utils.predict_fraud) y lo mapea al esquema crudo de FraudDNN.
"""
import json
import logging
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
import torch

from .fraud_dnn import FraudDNN

logger = logging.getLogger(__name__)

# --- SHIM de compatibilidad scikit-learn 1.6.x -> 1.7.x ---
try:
    import sklearn.compose._column_transformer as _ctmod  # type: ignore
    if not hasattr(_ctmod, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ctmod._RemainderColsList = _RemainderColsList
except Exception:
    pass

ML_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH   = ML_DIR / "fraud_dnn_weights.pt"
HPARAMS_PATH   = ML_DIR / "fraud_dnn_hparams.json"
EMBSPEC_PATH   = ML_DIR / "fraud_dnn_embedding_spec.json"
META_PATH      = ML_DIR / "fraud_dnn_meta.json"
FEATSFIN_PATH  = ML_DIR / "fraud_dnn_features_final.json"
PREPROC_PATH   = ML_DIR / "fraud_dnn_preprocessor.joblib"
BG_PATH        = ML_DIR / "fraud_dnn_background.npy"

DEVICE = torch.device("cpu")
SHAP_BG_SIZE = 50
SHAP_NSAMPLES = 100

# Valor por defecto para método de envío cuando el payload no lo trae.
DEFAULT_METODO_ENVIO = "Envío gratuito"

# ── Estado global (carga lazy) ───────────────────────────────────────
_model = None
_preproc = None
_emb_spec = None
_meta = None
_features_final = None
_num_idx = None
_hc_idx = None
_emb_keys = None
_background = None
_cat_lookup = None          # {col_raw: {valor_normalizado: valor_original}}
_explainer = None


def _strip(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.lower().strip()


def _ensure_loaded():
    global _model, _preproc, _emb_spec, _meta, _features_final
    global _num_idx, _hc_idx, _emb_keys, _background, _cat_lookup
    if _model is not None:
        return

    hp = json.loads(HPARAMS_PATH.read_text(encoding="utf-8"))
    _emb_spec = json.loads(EMBSPEC_PATH.read_text(encoding="utf-8"))
    _meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    _features_final = json.loads(FEATSFIN_PATH.read_text(encoding="utf-8"))
    _preproc = load(PREPROC_PATH)

    feats = _features_final["features"]
    high_card = _features_final["high_card"]
    _num_idx = [i for i, f in enumerate(feats) if f not in high_card]
    _hc_idx = {f: feats.index(f) for f in high_card}
    _emb_keys = list(_emb_spec.keys())

    model = FraudDNN(hp["n_numeric"], _emb_spec, hp["units"],
                     hp["dropout"], hp["n_residual_blocks"]).to(DEVICE)
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _model = model

    if BG_PATH.exists():
        _background = np.load(BG_PATH).astype(np.float32)

    # Lookups normalizados de categorías (one-hot + ordinal) desde el preprocesador
    _cat_lookup = {}
    try:
        ohe = _preproc.named_transformers_["ohe"]
        for col, cats in zip(_meta["low_card"], ohe.categories_):
            _cat_lookup[col] = {_strip(c): c for c in cats}
        ordc = _preproc.named_transformers_["ord"]
        for col, cats in zip(_meta["high_card"], ordc.categories_):
            _cat_lookup[col] = {_strip(c): c for c in cats}
    except Exception as e:  # pragma: no cover
        logger.warning("No se pudieron leer las categorías del preprocesador: %s", e)
        _cat_lookup = {}


def _match_cat(col, value):
    """Empareja un valor del payload con la categoría entrenada (case/acento-insensible).
    Para valores largos (p. ej. nombres de banco) aplica un respaldo por contención.
    Devuelve la categoría original si hay match; si no, el valor tal cual (el
    OneHotEncoder lo ignora y el OrdinalEncoder lo marca como desconocido)."""
    table = (_cat_lookup or {}).get(col, {})
    nv = _strip(value)
    if nv in table:
        return table[nv]
    if len(nv) >= 4:  # respaldo por contención (evita falsos positivos en códigos cortos)
        for norm_cat, original in table.items():
            if nv in norm_cat or norm_cat in nv:
                return original
    return value


# Departamentos cuyo código (3 letras del dataset) no coincide con las 3 primeras del nombre
_REGION_OVERRIDES = {
    "la libertad": "LAL", "la_libertad": "LAL", "lalibertad": "LAL",
    "huanuco": "HUC", "san martin": "SAM", "san_martin": "SAM",
    "madre de dios": "MDD", "moquegua": "MOQ", "pasco": "PAS",
    "puno": "PUN", "tumbes": "TUM", "apurimac": "APU",
}


def _region_code(value):
    """Convierte un nombre/código de región al código de 3 letras del dataset."""
    nv = _strip(value)
    code = _REGION_OVERRIDES.get(nv, nv.upper()[:3])
    return _match_cat("region_cliente", code)


# ── Mapeo payload de la app -> esquema crudo de FraudDNN ─────────────

def _brand(value):
    v = _strip(value)
    if "diner" in v:
        return _match_cat("marca", "dinersclub")
    return _match_cat("marca", v)


def _yesno(value):
    return "Si" if _strip(value) in ("si", "sí", "s", "yes", "y", "1", "true") else "No"


def _to_raw_df(payload: dict) -> pd.DataFrame:
    g = payload.get

    def num(k, default=0.0):
        try:
            return float(g(k))
        except (TypeError, ValueError):
            return float(default)

    tx_hour = int(num("tx_hour", 12))
    dow = int(num("tx_day_of_week", 0))
    weekend_night = 1 if (dow >= 5 or tx_hour <= 5 or tx_hour >= 23) else 0
    tiene_cupon = 1 if num("discount_amount", 0.0) > 0 else int(num("tiene_cupon", 0))

    row = {
        # numéricas
        "eci":                int(num("eci", 5)),
        "codigo_de_accion":   int(num("action_code", 6)),
        "importe_pedido":     num("transaction_amount", num("importe", 0.0)),
        "numero_de_cuotas":   int(num("num_installments", 0)),
        "cantidad_items":     int(num("num_items", 1)),
        "tiene_cupon":        tiene_cupon,
        "billing_mismatch":   int(num("billing_mismatch", 0)),
        "weekend_night_flag": weekend_night,
        "phone_format_valid": int(num("phone_format_valid", 1)),
        "tx_hour":            tx_hour,
        "C1_AAR":             num("AAR", 1.0),
        "C2_CMR":             num("CMR", 1.0),
        "C3_ASI":             num("ASI", 0.5),
        "C4_VRR":             num("VRR", 1.0),
        "C5_DAR":             num("DAR", 0.0),
        "C6_CSI":             num("CSI", 1.0),
        "C7_DPE":             num("DPE", 0.0),
        # one-hot baja cardinalidad
        "marca":              _brand(g("card_brand") or "visa"),
        "tipo_de_tarjeta":    _match_cat("tipo_de_tarjeta", g("card_type") or "credito"),
        "canal_pago":         _match_cat("canal_pago", g("payment_channel") or "Pago Web"),
        "metodo_envio":       _match_cat("metodo_envio", g("metodo_envio") or DEFAULT_METODO_ENVIO),
        "pago_con_yape":      _yesno(g("wallet_yape") or "no"),
        "pago_con_plin":      _yesno(g("wallet_plin") or "no"),
        # alta cardinalidad (embeddings)
        "emisor":             _match_cat("emisor", g("issuer_bank") or "bcp"),
        "region_cliente":     _region_code(g("customer_region") or "lima"),
        "categoria_producto": _match_cat("categoria_producto", str(g("product_category") or g("category") or "otros").title()),
    }
    return pd.DataFrame([row], columns=_meta["all_features"])


# ── Inferencia ───────────────────────────────────────────────────────

def _predict_proba(X_proc: np.ndarray) -> np.ndarray:
    """X_proc: array (n, n_features) en el espacio procesado. Devuelve prob de fraude."""
    X = np.atleast_2d(np.asarray(X_proc, dtype=np.float32)).copy()
    feats = _features_final["features"]
    for f, sp in _emb_spec.items():
        j = feats.index(f)
        X[:, j] = np.clip(np.round(X[:, j]), 0, sp["n_categories"])
    xn = torch.tensor(X[:, _num_idx], dtype=torch.float32, device=DEVICE)
    xe = torch.tensor(np.stack([X[:, _hc_idx[f]] for f in _emb_keys], axis=1),
                      dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        return _model(xn, xe).cpu().numpy().ravel()


def _transform(payload: dict) -> np.ndarray:
    df = _to_raw_df(payload)
    X = _preproc.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


# ── SHAP ─────────────────────────────────────────────────────────────

def _init_explainer():
    global _explainer
    if _explainer is not None or _background is None:
        return
    import shap
    bg = _background
    if bg.shape[0] > SHAP_BG_SIZE:
        rng = np.random.default_rng(42)
        bg = bg[rng.choice(bg.shape[0], SHAP_BG_SIZE, replace=False)]
    _explainer = shap.KernelExplainer(_predict_proba, bg)


def _shap_values(X_proc):
    try:
        _init_explainer()
        if _explainer is None:
            return None
        sv = _explainer.shap_values(X_proc, nsamples=SHAP_NSAMPLES, silent=True)
        sv = sv[0] if isinstance(sv, list) else sv
        return np.squeeze(np.asarray(sv))
    except Exception as e:
        logger.error("SHAP falló (%s) — se usa explicación por reglas.", e, exc_info=True)
        return None


# ── Nombres legibles, agregación y reglas de negocio ─────────────────

FEATURE_DISPLAY_NAMES = {
    "eci": "Código ECI (Autenticación)",
    "codigo_de_accion": "Código de Acción",
    "importe_pedido": "Monto del Pedido",
    "numero_de_cuotas": "Número de Cuotas",
    "cantidad_items": "Cantidad de Ítems",
    "tiene_cupon": "Tiene Cupón",
    "billing_mismatch": "Descuadre Facturación/Envío",
    "weekend_night_flag": "Fin de Semana / Noche",
    "phone_format_valid": "Formato de Teléfono Válido",
    "tx_hour": "Hora de la Transacción",
    "C1_AAR": "Ratio Monto/Promedio (AAR)",
    "C2_CMR": "Ratio Monto/Mediana (CMR)",
    "C3_ASI": "Índice Fortaleza Autenticación (ASI)",
    "C4_VRR": "Ratio de Velocidad (VRR)",
    "C5_DAR": "Ratio Denegación/Intentos (DAR)",
    "C6_CSI": "Índice Compartición Tarjeta (CSI)",
    "C7_DPE": "Entropía Patrón Denegación (DPE)",
    "marca": "Marca de Tarjeta",
    "tipo_de_tarjeta": "Tipo de Tarjeta",
    "canal_pago": "Canal de Pago",
    "metodo_envio": "Método de Envío",
    "pago_con_yape": "Pago con Yape",
    "pago_con_plin": "Pago con Plin",
    "emisor": "Banco Emisor",
    "region_cliente": "Región del Cliente",
    "categoria_producto": "Categoría de Producto",
}

# Base FraudDNN -> clave del payload de la app (para mostrar el valor y aplicar reglas)
_BASE_TO_PAYLOAD = {
    "C1_AAR": "AAR", "C2_CMR": "CMR", "C3_ASI": "ASI", "C4_VRR": "VRR",
    "C5_DAR": "DAR", "C6_CSI": "CSI", "C7_DPE": "DPE",
    "importe_pedido": "transaction_amount", "eci": "eci",
    "codigo_de_accion": "action_code", "tx_hour": "tx_hour",
    "numero_de_cuotas": "num_installments", "cantidad_items": "num_items",
    "emisor": "issuer_bank", "region_cliente": "customer_region",
    "categoria_producto": "product_category", "marca": "card_brand",
    "tipo_de_tarjeta": "card_type", "canal_pago": "payment_channel",
    "pago_con_yape": "wallet_yape", "pago_con_plin": "wallet_plin",
}


def _base_feature(name):
    for low in _meta["low_card"]:
        if name.startswith(low + "_"):
            return low
    return name


def _aggregate(final_names, sv):
    agg = {}
    for i, name in enumerate(final_names):
        base = _base_feature(name)
        agg[base] = agg.get(base, 0.0) + float(sv[i])
    return agg


def _build_business_explanation(ordered, payload):
    factors, reglas = [], []
    for base, impact in ordered:
        display = FEATURE_DISPLAY_NAMES.get(base, base.replace("_", " ").title())
        pkey = _BASE_TO_PAYLOAD.get(base, base)
        value = payload.get(pkey, "—")
        factors.append({"feature": base, "display_name": display,
                        "impact": float(impact), "value": value})

        if impact <= 0.02:
            continue
        if base == "C5_DAR" and float(payload.get("DAR", 0) or 0) > 0.3:
            reglas.append(f"Alto ratio de denegaciones ({float(payload.get('DAR', 0)):.2f}): "
                          "la tarjeta tiene un historial significativo de intentos rechazados.")
        elif base == "C4_VRR" and float(payload.get("VRR", 0) or 0) > 30:
            reglas.append(f"Velocidad de transacciones anormalmente alta (VRR={float(payload.get('VRR', 0)):.1f}): "
                          "patrón compatible con card testing automatizado.")
        elif base == "C1_AAR" and float(payload.get("AAR", 0) or 0) > 2.0:
            reglas.append(f"El monto supera el promedio del cliente (AAR={float(payload.get('AAR', 0)):.2f}).")
        elif base == "C2_CMR" and float(payload.get("CMR", 0) or 0) > 3.0:
            reglas.append(f"Monto muy por encima de la mediana de su categoría (CMR={float(payload.get('CMR', 0)):.2f}).")
        elif base == "C6_CSI" and float(payload.get("CSI", 0) or 0) > 1:
            reglas.append(f"La tarjeta está siendo compartida por {int(float(payload.get('CSI', 0)))} usuarios/regiones distintas.")
        elif base == "C7_DPE" and float(payload.get("DPE", 0) or 0) > 0.5:
            reglas.append("Alta entropía en los patrones de denegación: múltiples razones de rechazo (probing sistemático).")
        elif base == "C3_ASI" and float(payload.get("ASI", 1) or 0) < 0.3:
            reglas.append("Historial débil de autenticación segura (ASI bajo): el cliente raramente usa 3D Secure.")
        elif base == "eci" and int(float(payload.get("eci", 5) or 5)) not in (2, 5):
            reglas.append("La transacción carece de verificación 3D Secure (ECI de baja autenticación).")
        elif base == "weekend_night_flag":
            reglas.append("La transacción se realizó en horario/día atípico (noche o fin de semana).")
        elif base == "importe_pedido":
            reglas.append("El monto del pedido representa un volumen inusual o de alto riesgo.")
        elif base == "codigo_de_accion":
            reglas.append("El código de acción de la pasarela indica un patrón de respuesta atípico.")
        elif base == "emisor":
            reglas.append("El banco emisor muestra correlación con patrones de fraude recientes.")
        elif base == "categoria_producto":
            reglas.append(f"La categoría de producto ({payload.get('product_category', '—')}) presenta vulnerabilidad a fraude.")
        elif base == "region_cliente":
            reglas.append("La región del cliente cruza con zonas de alta incidencia de fraude.")
        elif base in ("marca", "tipo_de_tarjeta"):
            reglas.append("El tipo o marca de tarjeta está correlacionado con esquemas de fraude detectados.")

    reglas = list(dict.fromkeys(reglas))
    return factors, reglas


def _rule_based_fallback(payload: dict) -> dict:
    """Contribuciones aproximadas por heurística cuando SHAP no está disponible."""
    g = lambda k, d=0.0: float(payload.get(k, d) or d)
    contrib = {}
    aar = g("AAR", 1.0); contrib["C1_AAR"] = min((aar - 1.0) * 0.05, 0.3) if aar > 1.5 else 0.0
    cmr = g("CMR", 1.0); contrib["C2_CMR"] = min((cmr - 1.0) * 0.04, 0.25) if cmr > 2.0 else 0.0
    vrr = g("VRR", 1.0); contrib["C4_VRR"] = min((vrr - 1.0) * 0.03, 0.3) if vrr > 5.0 else 0.0
    dar = g("DAR", 0.0); contrib["C5_DAR"] = dar * 0.2 if dar > 0.2 else 0.0
    asi = g("ASI", 0.5); contrib["C3_ASI"] = (0.5 - asi) * 0.1 if asi < 0.3 else 0.0
    csi = g("CSI", 1.0); contrib["C6_CSI"] = (csi - 1.0) * 0.05 if csi > 1 else 0.0
    dpe = g("DPE", 0.0); contrib["C7_DPE"] = dpe * 0.08 if dpe > 0.5 else 0.0
    eci = int(g("eci", 5)); contrib["eci"] = 0.06 if eci not in (2, 5) else -0.02
    amt = g("transaction_amount", 0.0); contrib["importe_pedido"] = min(amt / 10000, 0.15) if amt > 500 else 0.0
    return contrib


# ── API pública ──────────────────────────────────────────────────────

def predict_and_explain(payload: dict, top_k: int = 6, aggregate: bool = True):
    """Recibe el payload crudo de la app y devuelve (score 0-100, explanation dict).

    explanation = {top_factors:[{feature,display_name,impact,value}],
                   explicacion_negocio:[str], sum_abs, prob}
    """
    _ensure_loaded()

    X_proc = _transform(payload)
    prob = float(_predict_proba(X_proc)[0])
    score = float(prob * 100)

    sv = _shap_values(X_proc)
    if sv is not None:
        contrib = _aggregate(_features_final["features"], sv) if aggregate else \
            {f: float(sv[i]) for i, f in enumerate(_features_final["features"])}
        ordered = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        factors, reglas = _build_business_explanation(ordered, payload)
        explanation = {
            "top_factors": factors,
            "explicacion_negocio": reglas,
            "sum_abs": float(sum(abs(v) for v in contrib.values())),
            "prob": round(prob, 6),
        }
    else:
        contrib = _rule_based_fallback(payload)
        ordered = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        factors, reglas = _build_business_explanation(ordered, payload)
        explanation = {
            "top_factors": factors,
            "explicacion_negocio": reglas,
            "sum_abs": float(sum(abs(v) for v in contrib.values())),
            "prob": round(prob, 6),
            "shap_fallback": True,
        }

    return score, explanation
