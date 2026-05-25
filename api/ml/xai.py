# api/ml/xai.py
"""
Módulo de Explicabilidad (XAI) para el modelo de detección de fraude.
Usa SHAP (KernelExplainer) para generar explicaciones de las predicciones
del modelo DAFD-Net (DNN multi-input con embeddings categoriales).

Modelo: DNN_optimized.keras (DAFD-Net — Entrenamiento/04_train_optimized.py)
Features numéricas: transaction_amount, discount_amount, tx_hour, tx_day_of_week,
    tx_month, eci, action_code, num_installments, num_items, AAR, CMR, ASI, VRR,
    DAR, CSI, DPE
Categorías OHE (low-card): card_brand, card_type, currency, transaction_status,
    payment_channel, wallet_yape, wallet_plin, product_category
Categorías Embedding (high-card): issuer_bank, customer_region, email_domain,
    denial_reason, bin
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

# Archivos del nuevo modelo DAFD-Net
MODEL_PATH        = ML_DIR / "best_model.keras"
DENSE_PREPROC     = ML_DIR / "dnn_dense_preprocessor.joblib"
ORD_ENCODER       = ML_DIR / "dnn_ordinal_encoder.joblib"
EMBED_SPEC_PATH   = ML_DIR / "dnn_embedding_spec.json"
FULL_PREPROC_PATH = ML_DIR / "preprocessor.joblib"
FEATS_PATH        = ML_DIR / "feature_names.json"
BG_PATH           = ML_DIR / "background.npy"
GROUPMAP_PATH     = ML_DIR / "group_map.json"

# ── Features del modelo ─────────────────────────────────────────────
NUMERIC_FEATURES = [
    "transaction_amount", "discount_amount",
    "tx_hour", "tx_day_of_week", "tx_month",
    "eci", "action_code", "num_installments", "num_items",
    "AAR", "CMR", "ASI", "VRR", "DAR", "CSI", "DPE",
]
OHE_CATEGORICAL_FEATURES = [
    "card_brand", "card_type", "currency",
    "transaction_status", "payment_channel",
    "wallet_yape", "wallet_plin",
    "product_category",
]
EMBED_COLS = ["issuer_bank", "customer_region", "email_domain", "denial_reason", "bin"]
ALL_FEATURES = NUMERIC_FEATURES + OHE_CATEGORICAL_FEATURES + EMBED_COLS

# Carga lazy (en la primera llamada)
_model = None
_dense_preproc = None
_ord_encoder = None
_embed_spec = None
_full_preproc = None
_feature_names = None
_background = None
_group_map = None
_explainer = None


def _ensure_artifacts():
    """Carga artefactos y el explainer una sola vez (lazy)."""
    global _model, _dense_preproc, _ord_encoder, _embed_spec
    global _full_preproc, _feature_names, _background, _group_map, _explainer

    if _model is not None:
        return

    # 1) Cargar artefactos
    _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    _dense_preproc = load(DENSE_PREPROC)
    _ord_encoder = load(ORD_ENCODER)
    with open(EMBED_SPEC_PATH, encoding="utf-8") as f:
        _embed_spec = json.load(f)
    _full_preproc = load(FULL_PREPROC_PATH)
    with open(FEATS_PATH, encoding="utf-8") as f:
        _feature_names = json.load(f)
    _background = np.load(BG_PATH)
    if GROUPMAP_PATH.exists():
        with open(GROUPMAP_PATH, encoding="utf-8") as f:
            _group_map = json.load(f)


def _build_multi_inputs(df):
    """Transforma un DataFrame de 1 fila en los inputs del modelo multi-input.
    Returns: list [X_dense, cat_col_1, cat_col_2, ...]
    """
    X_dense = _dense_preproc.transform(df)
    if hasattr(X_dense, "toarray"):
        X_dense = X_dense.toarray()

    X_ord = _ord_encoder.transform(df[EMBED_COLS])
    X_ord = np.clip(X_ord.astype(np.int64) + 1, 0, None)  # shift for embedding (0=padding)

    inputs = [X_dense]
    for j in range(len(EMBED_COLS)):
        inputs.append(X_ord[:, j:j+1])
    return inputs


def _predict_from_full_ohe(X_full):
    """Wrapper para SHAP: recibe la representación OHE completa (del full preprocessor)
    y genera predicción usando el modelo multi-input.

    Para SHAP necesitamos una función que tome un array numpy (background) y
    devuelva probabilidades. Aquí re-mapeamos del OHE completo al multi-input.

    NOTA: Dado que reconstruir desde OHE→raw→multi-input es costoso y propenso
    a errores de inversión, usamos una aproximación: el background ya está en
    formato OHE completo, y usamos el full preprocessor para SHAP directamente.
    """
    return _model.predict(
        _split_ohe_to_multi(X_full), verbose=0
    ).ravel()


def _split_ohe_to_multi(X_full):
    """Divide la representación OHE completa en dense + ordinal para el modelo.

    El preprocessor completo produce: [num_scaled | ohe_all_cats]
    El dense_preprocessor produce:    [num_scaled | ohe_low_cats]
    Necesitamos mapear de uno al otro.

    Estrategia simplificada: como SHAP opera sobre el espacio OHE completo,
    re-usamos las columnas numéricas + OHE low-card del full preprocessor,
    y para las columnas de embedding, tomamos el argmax de las columnas OHE
    correspondientes a cada categoría.
    """
    n_samples = X_full.shape[0]

    # Posiciones en el full preprocessor output
    n_num = len(NUMERIC_FEATURES)

    # Extraer numéricas (primeras n_num columnas)
    X_num = X_full[:, :n_num]

    # Extraer OHE para low-card categories
    # El full preprocessor hace OHE de ALL cats (OHE + EMBED), pero
    # el dense preprocessor solo hace OHE de OHE cats.
    # Las columnas del full preprocessor son: [num | ohe_of_all_cats]
    # Necesitamos identificar qué columnas corresponden a las OHE low-card cats

    # Obtener los nombres de features del full preprocessor
    try:
        full_cat_names = list(_full_preproc.named_transformers_["cat"]
                            .named_steps["onehot"]
                            .get_feature_names_out(OHE_CATEGORICAL_FEATURES + EMBED_COLS))
    except Exception:
        full_cat_names = _feature_names[n_num:]

    # Identificar columnas que son de OHE low-card
    ohe_low_indices = []
    embed_groups = {col: [] for col in EMBED_COLS}

    for idx, fname in enumerate(full_cat_names):
        is_embed = False
        for ecol in EMBED_COLS:
            if fname.startswith(ecol + "_"):
                embed_groups[ecol].append(idx)
                is_embed = True
                break
        if not is_embed:
            ohe_low_indices.append(idx)

    # Dense input = num + ohe_low
    X_ohe_low = X_full[:, [n_num + i for i in ohe_low_indices]]
    X_dense = np.hstack([X_num, X_ohe_low])

    # Ordinal inputs (argmax de cada grupo OHE → índice de embedding)
    cat_inputs = []
    for ecol in EMBED_COLS:
        indices = embed_groups[ecol]
        if indices:
            ohe_slice = X_full[:, [n_num + i for i in indices]]
            ordinal = np.argmax(ohe_slice, axis=1).reshape(-1, 1).astype(np.int32) + 1
        else:
            ordinal = np.zeros((n_samples, 1), dtype=np.int32)
        cat_inputs.append(ordinal)

    return [X_dense] + cat_inputs


def _init_explainer():
    """KernelExplainer sobre el espacio OHE completo."""
    global _explainer
    if _explainer is not None:
        return

    bg = _background
    # Submuestrear si background es muy grande
    if bg.shape[0] > 100:
        rng = np.random.default_rng(42)
        idx = rng.choice(bg.shape[0], 100, replace=False)
        bg = bg[idx]

    _explainer = shap.KernelExplainer(
        _predict_from_full_ohe, bg
    )


def _transform_full(payload: dict) -> np.ndarray:
    """Transforma el payload crudo usando el preprocessor completo (para SHAP)."""
    df = pd.DataFrame([payload])
    X = _full_preproc.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X


# ── Mapeo de nombres legibles para explicabilidad ────────────────────
FEATURE_DISPLAY_NAMES = {
    "transaction_amount": "Monto Transacción",
    "transaction": "Monto Transacción",
    "discount_amount": "Monto Descuento",
    "discount": "Monto Descuento",
    "tx_hour": "Hora",
    "tx_day_of_week": "Día de Semana",
    "tx_month": "Mes",
    "tx": "Temporal (Hora/Día/Mes)",
    "eci": "Código ECI (Autenticación)",
    "action_code": "Código de Acción",
    "action": "Código de Acción",
    "num_installments": "Cuotas",
    "num_items": "Cantidad de Ítems",
    "num": "Cuotas / Ítems",
    "AAR": "Ratio Monto/Promedio (AAR)",
    "CMR": "Ratio Monto/Mediana Categoría (CMR)",
    "ASI": "Índice Fortaleza Autenticación (ASI)",
    "VRR": "Ratio Velocidad Riesgo (VRR)",
    "DAR": "Ratio Denegación/Intentos (DAR)",
    "CSI": "Índice Compartición Tarjeta (CSI)",
    "DPE": "Entropía Patrón Denegación (DPE)",
    "card_brand": "Marca de Tarjeta",
    "card": "Tarjeta (Marca/Tipo)",
    "card_type": "Tipo de Tarjeta",
    "currency": "Moneda",
    "transaction_status": "Estado Transacción",
    "payment_channel": "Canal de Pago",
    "payment": "Canal de Pago",
    "wallet_yape": "Wallet Yape",
    "wallet_plin": "Wallet Plin",
    "wallet": "Wallets Digitales",
    "product_category": "Categoría Producto",
    "product": "Categoría Producto",
    "issuer_bank": "Banco Emisor",
    "issuer": "Banco Emisor",
    "customer_region": "Región del Cliente",
    "customer": "Región del Cliente",
    "email_domain": "Dominio Email",
    "email": "Dominio Email",
    "denial_reason": "Razón de Denegación",
    "denial": "Razón de Denegación",
    "bin": "BIN de Tarjeta",
}


def _aggregate(feature_contrib: dict) -> dict:
    """
    Agrupa impactos One-Hot por feature base usando el group_map.
    """
    if not _group_map:
        # Fallback: agrupar por prefijo
        categories = [
            "card_brand", "card_type", "currency",
            "transaction_status", "payment_channel",
            "wallet_yape", "wallet_plin",
            "product_category", "issuer_bank",
            "customer_region", "email_domain",
            "denial_reason", "bin",
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

    # Usar group_map: invertir para lookup rápido
    feat_to_group = {}
    for group, feats in _group_map.items():
        for f in feats:
            feat_to_group[f] = group

    agg = {}
    for name, val in feature_contrib.items():
        group = feat_to_group.get(name, name)
        agg[group] = agg.get(group, 0.0) + val
    return agg


def predict_and_explain(payload: dict, top_k: int = 6, aggregate: bool = True):
    """
    Recibe un dict con las features crudas y retorna (score, explanation).

    payload debe contener claves coincidentes con ALL_FEATURES:
      transaction_amount, discount_amount, tx_hour, tx_day_of_week, tx_month,
      eci, action_code, num_installments, num_items,
      AAR, CMR, ASI, VRR, DAR, CSI, DPE,
      card_brand, card_type, currency, transaction_status, payment_channel,
      wallet_yape, wallet_plin, product_category,
      issuer_bank, customer_region, email_domain, denial_reason, bin
    """
    _ensure_artifacts()

    # 1. Predicción con modelo multi-input
    df = pd.DataFrame([payload])
    inputs = _build_multi_inputs(df)
    prob = float(_model.predict(inputs, verbose=0).ravel()[0])
    score = float(prob * 100)

    # 2. SHAP values sobre representación OHE completa
    _init_explainer()
    X_full = _transform_full(payload)
    shap_vals = _explainer.shap_values(X_full, nsamples=100)
    sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
    sv = np.squeeze(sv)

    contrib = {name: float(sv[i]) for i, name in enumerate(_feature_names)}

    if aggregate:
        contrib = _aggregate(contrib)

    ordered = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]

    # 3. Añadir nombres legibles y lógica de negocio
    factors = []
    explicacion_negocio = []

    for k, v in ordered:
        display = FEATURE_DISPLAY_NAMES.get(k, k.replace("_", " ").title())
        val_raw = payload.get(k, "—")
        factors.append({"feature": k, "display_name": display, "impact": v, "value": val_raw})

        # Generar reglas de negocio para factores agravantes
        if v > 0.03:
            if k in ("DAR", "denial") and payload.get("DAR", 0) > 0.3:
                explicacion_negocio.append(
                    f"Alto ratio de denegaciones ({payload.get('DAR', 0):.2f}): "
                    "la tarjeta tiene un historial significativo de intentos rechazados."
                )
            elif k in ("VRR", ) and payload.get("VRR", 0) > 30:
                explicacion_negocio.append(
                    f"Velocidad de transacciones anormalmente alta (VRR={payload.get('VRR', 0):.1f}): "
                    "patrón compatible con card testing automatizado."
                )
            elif k in ("AAR", ) and payload.get("AAR", 0) > 2.0:
                explicacion_negocio.append(
                    f"El monto supera significativamente el promedio del cliente (AAR={payload.get('AAR', 0):.2f})."
                )
            elif k in ("CMR", ) and payload.get("CMR", 0) > 3.0:
                explicacion_negocio.append(
                    f"Monto muy por encima de la mediana de su categoría (CMR={payload.get('CMR', 0):.2f})."
                )
            elif k in ("CSI", ) and payload.get("CSI", 0) > 1:
                explicacion_negocio.append(
                    f"La tarjeta está siendo compartida por {int(payload.get('CSI', 0))} usuarios distintos."
                )
            elif k in ("DPE", ) and payload.get("DPE", 0) > 0.5:
                explicacion_negocio.append(
                    "Se detectó alta entropía en los patrones de denegación: "
                    "múltiples razones de rechazo distintas sugieren probing sistemático."
                )
            elif k in ("ASI", ) and payload.get("ASI", 0) < 0.3:
                explicacion_negocio.append(
                    "Historial débil de autenticación segura (ASI bajo): "
                    "el cliente raramente usa 3D Secure."
                )
            elif k in ("transaction", "transaction_amount"):
                explicacion_negocio.append(
                    "El monto transaccional neto representa un volumen inusual o de alto riesgo."
                )
            elif k in ("eci", ) and payload.get("eci", 5) not in (2, 5):
                explicacion_negocio.append(
                    "La transacción carece de verificación 3D Secure (ECI indica baja autenticación)."
                )
            elif k in ("tx", ) or k == "tx_hour":
                hour = payload.get("tx_hour", 12)
                if hour <= 5 or hour >= 23:
                    explicacion_negocio.append(
                        "La transacción se realizó en horario atípico (madrugada/noche)."
                    )
            elif k in ("email", "email_domain"):
                domain = str(payload.get("email_domain", ""))
                temp_domains = {"10minutemail.com", "guerrillamail.com", "mailinator.com",
                                "tempmail.com", "yopmail.com"}
                if domain.lower() in temp_domains:
                    explicacion_negocio.append(
                        f"El email usa un dominio temporal/desechable ({domain}), "
                        "frecuentemente asociado a fraude."
                    )
            elif k in ("card", "card_brand", "card_type"):
                explicacion_negocio.append(
                    "El tipo o marca de tarjeta está correlacionado con esquemas de fraude detectados."
                )
            elif k in ("issuer", "issuer_bank"):
                explicacion_negocio.append(
                    "El banco emisor muestra correlación con patrones de fraude recientes."
                )
            elif k in ("product", "product_category"):
                cat = str(payload.get("product_category", "")).replace("_", " ").title()
                explicacion_negocio.append(
                    f"La categoría de producto ({cat}) presenta alta vulnerabilidad a fraude."
                )
            elif k in ("customer", "customer_region"):
                explicacion_negocio.append(
                    "La región del cliente cruza con zonas de alta incidencia de fraude."
                )
            elif k in ("action", "action_code"):
                explicacion_negocio.append(
                    "El código de acción de la pasarela indica un patrón de respuesta atípico."
                )
            elif k in ("wallet", "wallet_yape", "wallet_plin"):
                explicacion_negocio.append(
                    "El uso (o no uso) de wallet digital influye en el perfil de riesgo."
                )

    # Deduplicar
    explicacion_negocio = list(dict.fromkeys(explicacion_negocio))

    explanation = {
        "top_factors": factors,
        "explicacion_negocio": explicacion_negocio,
        "sum_abs": float(sum(abs(v) for v in contrib.values())),
        "prob": round(prob, 6),
    }
    return score, explanation
