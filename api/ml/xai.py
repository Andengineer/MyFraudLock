# api/ml/xai.py
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
    # Si por algún motivo no existe el módulo interno, seguimos;
    # el objetivo es solo añadir el atributo faltante cuando aplica.
    pass

ML_DIR = Path(__file__).resolve().parent
MODEL_PATH = ML_DIR / "dnn_best.keras"
PREPROC_PATH = ML_DIR / "preprocessor.joblib"
FEATS_PATH = ML_DIR / "feature_names.json"
BG_PATH = ML_DIR / "background.npy"
GROUPMAP_PATH = ML_DIR / "group_map.json"

# Carga lazy (en la primera llamada), para que runserver no crashee al importar el módulo
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
        _explainer = shap.KernelExplainer(lambda X: _model.predict(X, verbose=0), _background)

def _transform(payload: dict) -> np.ndarray:
    # payload debe traer TODAS las columnas crudas que espera el preprocesador
    df = pd.DataFrame([payload])
    X = _preproc.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X

def _aggregate(feature_contrib: dict) -> dict:
    # Agrupa impactos por feature base usando group_map si existe; si no, por prefijo antes del "_"
    if not _group_map:
        agg = {}
        for name, val in feature_contrib.items():
            base = name.split("_")[0]
            agg[base] = agg.get(base, 0.0) + val
        return agg
    agg, used = {}, set()
    for base, cols in _group_map.items():
        s = sum(feature_contrib.get(c, 0.0) for c in cols)
        if s != 0.0:
            agg[base] = agg.get(base, 0.0) + s
        used.update(cols)
    for name, val in feature_contrib.items():
        if name not in used:
            agg[name] = agg.get(name, 0.0) + val
    return agg

def predict_and_explain(payload: dict, top_k: int = 6, aggregate: bool = True):
    """
    payload DEBE contener estas claves crudas (con tus nombres confirmados):
      ['amt','amt_log1p','age','hour','weekday','month','is_weekend','city_pop','category','state','gender']
    """
    _ensure_artifacts()

    X = _transform(payload)
    prob = float(_model.predict(X, verbose=0).ravel()[0])
    score = float(prob * 100)

    shap_vals = _explainer.shap_values(X)
    sv = shap_vals[0][0] if isinstance(shap_vals, list) else shap_vals[0]
    sv = np.squeeze(sv)  # (74,1) → (74,) or keep (74,) as-is
    contrib = {name: float(sv[i]) for i, name in enumerate(_feature_names)}
    if aggregate:
        contrib = _aggregate(contrib)
    ordered = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    explanation = {
        "top_factors": [{"feature": k, "impact": v} for k, v in ordered],
        "sum_abs": float(sum(abs(v) for v in contrib.values())),
        "prob": round(prob, 6)  # <-- agrega esto para ver si el modelo satura
    }
    return score, explanation
