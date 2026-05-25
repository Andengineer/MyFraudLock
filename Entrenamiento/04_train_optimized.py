#!/usr/bin/env python3
"""
04_train_optimized.py — Fase de Optimización de Hiperparámetros
======================================================================
Lee el resultado de la Fase 1 (03_balance_experiment.py):
  - Mejor técnica de balanceo
  - 5 modelos base

Flujo:
  1. Entrena los 5 modelos con hiperparámetros DEFAULT usando el mejor balanceo
  2. Optimiza hiperparámetros con Optuna (20 trials) para cada modelo
  3. Compara Default vs Optimizado
  4. Exporta el modelo ganador + SHAP

NOTA: Si el ganador global es XGBoost, se documenta pero se exporta el
      mejor modelo de Deep Learning para compatibilidad con el backend Django.
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json, time, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import OrderedDict

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, confusion_matrix, matthews_corrcoef,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from joblib import dump
import xgboost as xgb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ─── Config ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "outputs" / "data"
MODEL_DIR  = BASE_DIR / "outputs" / "models"
FIG_DIR    = BASE_DIR / "outputs" / "figures"
XAI_DIR    = BASE_DIR / "outputs" / "explainability"
for d in [MODEL_DIR, FIG_DIR, XAI_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.grid": True, "grid.alpha": 0.3,
})

OPTUNA_TRIALS = 15  # Trials per model (uniform across baselines)
# Per-model trial budget (15 for all by default — adjust here if compute-constrained)
OPTUNA_TRIALS_PER_MODEL = {
    "DNN":             15,
    "CNN_1D":          15,
    "RNN_GRU":         15,
    "AutoEncoder_Clf": 15,
    "XGBoost":         15,
}

# ═══════════════════════════════════════════════════════════════════════
# FEATURES & LABELS
# ═══════════════════════════════════════════════════════════════════════
NUMERIC_FEATURES = [
    # Raw transactional fields
    "transaction_amount", "discount_amount",
    "tx_hour", "tx_day_of_week", "tx_month",
    "eci", "action_code", "num_installments", "num_items",
    # 7 transactional ratios from §3.C.2 (4 Vasquez-analog + 3 novel)
    "AAR", "CMR", "ASI", "VRR", "DAR", "CSI", "DPE",
]
# Categoricals: low-cardinality use OHE (all models), high-cardinality use
# embeddings (DNN only; baselines use OHE for fair comparison).
OHE_CATEGORICAL_FEATURES = [
    "card_brand", "card_type", "currency",
    "transaction_status", "payment_channel",
    "wallet_yape", "wallet_plin",
    "product_category",
]
EMBED_CATEGORICAL_FEATURES = {
    "issuer_bank":     6,
    "customer_region": 6,
    "email_domain":    5,
    "denial_reason":   5,
    "bin":             8,
}
EMBED_COLS = list(EMBED_CATEGORICAL_FEATURES.keys())
CATEGORICAL_FEATURES = OHE_CATEGORICAL_FEATURES + EMBED_COLS

TARGET = "is_fraud_ground_truth"
HEURISTIC_BASELINE_COL = "heuristic_label"

MODEL_COLORS = {
    "DNN": "#3498db", "CNN_1D": "#e74c3c", "RNN_GRU": "#2ecc71",
    "AutoEncoder_Clf": "#9b59b6", "XGBoost": "#f39c12",
}
MODEL_LABELS = {
    "DNN": "DNN", "CNN_1D": "CNN-1D", "RNN_GRU": "RNN-GRU",
    "AutoEncoder_Clf": "Autoencoder", "XGBoost": "XGBoost",
}
BALANCE_LABELS = {
    "baseline": "Sin Balanceo", "smote_tomek": "SMOTE-Tomek", "adasyn": "ADASYN",
}


# ═══════════════════════════════════════════════════════════════════════
# 1. CARGA
# ═══════════════════════════════════════════════════════════════════════

def apply_balancing_dnn_emb(X_dense, X_ord, y, method):
    """SMOTENC-based balancing for DAFD-Net multi-input (preserves valid
    ordinal indices required by Embedding layers).

    - baseline:    pass-through
    - smote_tomek: SMOTENC + TomekLinks cleanup
    - adasyn:      SMOTENC with denser sampling_strategy (ADASYN has no NC variant)
    """
    if method == "baseline":
        return X_dense, X_ord, y

    n_dense = X_dense.shape[1]
    n_ord   = X_ord.shape[1]
    X_combined = np.hstack([X_dense, X_ord.astype(np.float64)])
    cat_indices = list(range(n_dense, n_dense + n_ord))

    print(f"\n  [DNN-Emb] Aplicando SMOTENC ({BALANCE_LABELS[method]})...")
    if method == "smote_tomek":
        smote_nc = SMOTENC(categorical_features=cat_indices,
                            random_state=SEED, sampling_strategy=0.4, k_neighbors=5)
        X_bal, y_bal = smote_nc.fit_resample(X_combined, y)
        tomek = TomekLinks(sampling_strategy="majority")
        X_bal, y_bal = tomek.fit_resample(X_bal, y_bal)
    elif method == "adasyn":
        smote_nc = SMOTENC(categorical_features=cat_indices,
                            random_state=SEED, sampling_strategy=0.5, k_neighbors=5)
        X_bal, y_bal = smote_nc.fit_resample(X_combined, y)
    else:
        raise ValueError(f"Método desconocido: {method}")

    X_dense_bal = X_bal[:, :n_dense]
    X_ord_bal   = np.round(X_bal[:, n_dense:]).astype(np.int64)
    for j in range(n_ord):
        max_val = X_ord[:, j].max()
        X_ord_bal[:, j] = np.clip(X_ord_bal[:, j], 0, max_val)
    return X_dense_bal, X_ord_bal, y_bal


def load_and_prepare(best_balance):
    """Carga datos, preprocesa con dual representation (OHE para baselines +
    Ordinal para DNN embeddings) y aplica el mejor balanceo."""
    print("\n[1] CARGA Y PREPROCESAMIENTO")
    print("-" * 50)

    df = pd.read_csv(DATA_DIR / "fraud_ecommerce_dataset_processed.csv")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].values

    # Full OHE for non-DNN baselines (CNN, RNN, AE, XGBoost)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SKPipeline([("scaler", StandardScaler())]), NUMERIC_FEATURES),
            ("cat", SKPipeline([("onehot", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            ))]), CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )
    # DNN dense stream: numerics + OHE of LOW-card cats
    dnn_dense_preprocessor = ColumnTransformer(
        transformers=[
            ("num", SKPipeline([("scaler", StandardScaler())]), NUMERIC_FEATURES),
            ("ohe_low", SKPipeline([("onehot", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            ))]), OHE_CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )
    dnn_ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=SEED, stratify=y_temp
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc   = preprocessor.transform(X_val)
    X_test_proc  = preprocessor.transform(X_test)

    # DNN-Embedding matrices
    X_train_dense = dnn_dense_preprocessor.fit_transform(X_train)
    X_val_dense   = dnn_dense_preprocessor.transform(X_val)
    X_test_dense  = dnn_dense_preprocessor.transform(X_test)
    X_train_ord = dnn_ordinal_encoder.fit_transform(X_train[EMBED_COLS]).astype(np.int64)
    X_val_ord   = dnn_ordinal_encoder.transform(X_val[EMBED_COLS]).astype(np.int64)
    X_test_ord  = dnn_ordinal_encoder.transform(X_test[EMBED_COLS]).astype(np.int64)
    cardinalities = [int(X_train_ord[:, j].max()) + 2 for j in range(len(EMBED_COLS))]
    X_train_ord = X_train_ord + 1
    X_val_ord   = np.clip(X_val_ord + 1, 0, None)
    X_test_ord  = np.clip(X_test_ord + 1, 0, None)

    num_names = NUMERIC_FEATURES
    cat_names = list(preprocessor.named_transformers_["cat"]
                     .named_steps["onehot"]
                     .get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = num_names + cat_names

    dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    dump(dnn_dense_preprocessor, MODEL_DIR / "dnn_dense_preprocessor.joblib")
    dump(dnn_ordinal_encoder, MODEL_DIR / "dnn_ordinal_encoder.joblib")
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(MODEL_DIR / "dnn_embedding_spec.json", "w") as f:
        json.dump({
            "embed_cols": EMBED_COLS,
            "embed_dims": list(EMBED_CATEGORICAL_FEATURES.values()),
            "cardinalities": cardinalities,
        }, f, indent=2)

    # ── Precompute ALL 3 balance variants (per-model lookup in training) ──
    def _apply_balance_ohe(method):
        if method == "baseline":
            return X_train_proc, y_train
        elif method == "smote_tomek":
            st = SMOTETomek(smote=SMOTE(random_state=SEED, sampling_strategy=0.4),
                             random_state=SEED)
            return st.fit_resample(X_train_proc, y_train)
        elif method == "adasyn":
            return ADASYN(random_state=SEED, sampling_strategy=0.4).fit_resample(
                X_train_proc, y_train)

    balanced_ohe = {}
    balanced_dnn = {}
    for method in ["baseline", "smote_tomek", "adasyn"]:
        Xb, yb = _apply_balance_ohe(method)
        balanced_ohe[method] = (Xb, yb)
        Xd, Xo, yd = apply_balancing_dnn_emb(X_train_dense, X_train_ord, y_train, method)
        balanced_dnn[method] = (Xd, Xo, yd)
        print(f"  Balance '{method}': OHE {Xb.shape[0]:,} rows | DNN-Emb {Xd.shape[0]:,} rows")

    # Globally-best balance (used as the fallback / overall winner reference)
    X_train_bal, y_train_bal = balanced_ohe[best_balance]
    X_train_dense_bal, X_train_ord_bal, y_train_bal_dnn = balanced_dnn[best_balance]

    print(f"  Train (global best): {len(X_train_bal):,} | Val: {len(X_val_proc):,} | Test: {len(X_test_proc):,}")
    print(f"  Features: {len(feature_names)}")
    print(f"  DNN-Emb dense_dim={X_train_dense.shape[1]} | cardinalities={cardinalities}")

    dnn_data = {
        "X_train_dense":     X_train_dense_bal,
        "X_val_dense":       X_val_dense,
        "X_test_dense":      X_test_dense,
        "X_train_ord":       X_train_ord_bal,
        "X_val_ord":         X_val_ord,
        "X_test_ord":        X_test_ord,
        "cardinalities":     cardinalities,
        "dense_dim":         X_train_dense.shape[1],
        "y_train":           y_train_bal_dnn,
        # Per-method variants for per-model balance lookup
        "balanced_dnn":      balanced_dnn,
    }
    return (X_train_bal, y_train_bal, X_val_proc, y_val,
            X_test_proc, y_test, feature_names, preprocessor, dnn_data,
            balanced_ohe)


# ═══════════════════════════════════════════════════════════════════════
# 2. MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(y_test, y_proba, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "auc_roc":     roc_auc_score(y_test, y_proba),
        "auc_pr":      average_precision_score(y_test, y_proba),
        "f1":          f1_score(y_test, y_pred),
        "precision":   precision_score(y_test, y_pred, zero_division=0),
        "recall":      recall_score(y_test, y_pred),
        "fpr":         fp / (fp + tn) if (fp + tn) > 0 else 0,
        "mcc":         matthews_corrcoef(y_test, y_pred),
        "g_mean":      np.sqrt(recall_score(y_test, y_pred) * specificity),
        "costo_negocio": fn * 50 + fp * 1,  # FN cuesta 50x más que FP en fraude real
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. BUILDERS DEFAULT
# ═══════════════════════════════════════════════════════════════════════

def _residual_block(x, units, dropout_rate=0.25):
    shortcut = x
    h = layers.Dense(units, activation="relu")(x)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(dropout_rate)(h)
    h = layers.Dense(units, activation="relu")(h)
    h = layers.BatchNormalization()(h)
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units, activation="linear")(shortcut)
    out = layers.Add()([shortcut, h])
    return layers.Activation("relu")(out)


def _se_block(x, ratio=4):
    units = x.shape[-1]
    se = layers.Dense(units // ratio, activation="relu")(x)
    se = layers.Dense(units, activation="sigmoid")(se)
    return layers.Multiply()([x, se])


def build_dnn_default(dense_dim, cardinalities=None, embed_dims=None, embed_cols=None):
    """DAFD-Net default: ResNet+SE+Embeddings (Camino B). When cardinalities
    is None, falls back to the legacy single-input architecture."""
    if cardinalities is None or not cardinalities:
        inp = layers.Input(shape=(dense_dim,))
        x = layers.Dense(384, activation="relu")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = _residual_block(x, 256, dropout_rate=0.30); x = _se_block(x)
        x = _residual_block(x, 128, dropout_rate=0.25); x = _se_block(x)
        x = _residual_block(x, 64,  dropout_rate=0.20)
        x = layers.Dense(32, activation="relu")(x); x = layers.Dropout(0.15)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        return Model(inp, out, name="DNN")

    num_input = layers.Input(shape=(dense_dim,), name="dense_in")
    cat_inputs, cat_embeds = [], []
    for col, cardinality, dim in zip(embed_cols, cardinalities, embed_dims):
        ci = layers.Input(shape=(1,), name=f"cat_{col}", dtype="int32")
        ce = layers.Embedding(cardinality, dim, name=f"emb_{col}")(ci)
        ce = layers.Flatten(name=f"flat_{col}")(ce)
        cat_inputs.append(ci); cat_embeds.append(ce)
    x = layers.Concatenate(name="fusion")([num_input] + cat_embeds)
    x = layers.Dense(384, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = _residual_block(x, 256, dropout_rate=0.30); x = _se_block(x)
    x = _residual_block(x, 128, dropout_rate=0.25); x = _se_block(x)
    x = _residual_block(x, 64,  dropout_rate=0.20)
    x = layers.Dense(32, activation="relu")(x); x = layers.Dropout(0.15)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model([num_input] + cat_inputs, out, name="DNN")

def build_cnn_default(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inp)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out, name="CNN_1D")

def build_gru_default(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inp)
    x = layers.GRU(64, return_sequences=True)(x)
    x = layers.GRU(32)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out, name="RNN_GRU")

def build_ae_default(input_dim):
    inp = layers.Input(shape=(input_dim,))
    enc = layers.Dense(128, activation="relu")(inp)
    enc = layers.BatchNormalization()(enc)
    enc = layers.Dropout(0.3)(enc)
    enc = layers.Dense(64, activation="relu")(enc)
    enc = layers.BatchNormalization()(enc)
    bottleneck = layers.Dense(24, activation="relu", name="bottleneck")(enc)
    dec = layers.Dense(64, activation="relu")(bottleneck)
    dec = layers.Dense(128, activation="relu")(dec)
    cls = layers.Dense(32, activation="relu")(bottleneck)
    cls = layers.Dropout(0.2)(cls)
    classification = layers.Dense(1, activation="sigmoid", name="classification")(cls)
    return Model(inp, classification, name="AutoEncoder_Clf")

def build_xgb_default(input_dim):
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="auc", random_state=SEED,
        use_label_encoder=False, verbosity=0, n_jobs=-1,
    )

DEFAULT_BUILDERS = OrderedDict([
    ("DNN", build_dnn_default),
    ("CNN_1D", build_cnn_default),
    ("RNN_GRU", build_gru_default),
    ("AutoEncoder_Clf", build_ae_default),
    ("XGBoost", build_xgb_default),
])


# ═══════════════════════════════════════════════════════════════════════
# 4. ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════

# Equal wall-clock budget across DL models (defensible against reviewers)
MAX_WALLCLOCK_SECONDS_DEFAULT = 600     # per default-model training
MAX_WALLCLOCK_SECONDS_TRIAL = 90        # per Optuna trial
PATIENCE_EPOCHS = 15
LR_PATIENCE_EPOCHS = 6
MAX_EPOCHS_HARD_CAP = 200


class WallClockBudget(keras.callbacks.Callback):
    """Stops training when elapsed wall-clock time exceeds the budget."""
    def __init__(self, budget_seconds: float):
        super().__init__()
        self.budget = budget_seconds
        self._start = None

    def on_train_begin(self, logs=None):
        self._start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if (time.time() - self._start) > self.budget:
            self.model.stop_training = True


def train_keras(model, X_train, y_train, X_val, y_val, name,
                lr=0.001, batch_size=256,
                wallclock_budget=MAX_WALLCLOCK_SECONDS_DEFAULT):
    """Trains a Keras model with equal wall-clock budget across models."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        EarlyStopping(monitor="val_auc", patience=PATIENCE_EPOCHS,
                      restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=LR_PATIENCE_EPOCHS, min_lr=1e-6),
        WallClockBudget(wallclock_budget),
    ]
    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS_HARD_CAP,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    train_time = time.time() - start
    best_ep = int(np.argmax(history.history.get("val_auc", [0])))
    print(f"    ✓ {name}: {train_time:.1f}s ({len(history.history.get('loss', []))} ep, best @ {best_ep+1})")
    return model, history, train_time


def train_xgb(model, X_train, y_train, X_val, y_val, name):
    n_neg, n_pos = np.sum(y_train == 0), np.sum(y_train == 1)
    model.set_params(scale_pos_weight=n_neg / n_pos)
    start = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.time() - start
    print(f"    ✓ {name}: {train_time:.1f}s")
    return model, None, train_time


def evaluate(model, X_test, y_test, is_xgb=False):
    if is_xgb:
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = compute_metrics(y_test, y_proba, y_pred)
    return metrics, y_proba, y_pred


# ═══════════════════════════════════════════════════════════════════════
# 5. OPTUNA OBJECTIVES
# ═══════════════════════════════════════════════════════════════════════

def make_dnn_objective(dense_dim, X_train_inputs, y_train, X_val_inputs, y_val,
                        cardinalities, embed_cols):
    """Optuna objective for DAFD-Net with categorical embeddings.

    X_train_inputs / X_val_inputs are LISTS: [dense, cat_col_1, cat_col_2, ...]
    Optuna tunes layer widths, dropout, learning rate, batch size, and the
    per-column embedding dimension scale factor.
    """
    def objective(trial):
        units_1 = trial.suggest_categorical("units_1", [256, 384, 512])
        units_2 = trial.suggest_categorical("units_2", [128, 192, 256])
        units_3 = trial.suggest_categorical("units_3", [64, 96, 128])
        dropout = trial.suggest_float("dropout", 0.15, 0.45, step=0.05)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        emb_scale = trial.suggest_categorical("emb_scale", [0.75, 1.0, 1.25, 1.5])

        # Per-column embedding dims (scaled around the configured defaults)
        base_dims = list(EMBED_CATEGORICAL_FEATURES.values())
        trial_dims = [max(2, int(round(d * emb_scale))) for d in base_dims]

        # Build DAFD-Net with embeddings
        num_input = layers.Input(shape=(dense_dim,), name="dense_in")
        cat_inputs, cat_embeds = [], []
        for col, card, dim in zip(embed_cols, cardinalities, trial_dims):
            ci = layers.Input(shape=(1,), name=f"cat_{col}", dtype="int32")
            ce = layers.Embedding(card, dim, name=f"emb_{col}")(ci)
            ce = layers.Flatten(name=f"flat_{col}")(ce)
            cat_inputs.append(ci); cat_embeds.append(ce)
        x = layers.Concatenate(name="fusion")([num_input] + cat_embeds)
        x = layers.Dense(units_1, activation="relu")(x)
        x = layers.BatchNormalization()(x); x = layers.Dropout(dropout)(x)
        x = _residual_block(x, units_2, dropout_rate=dropout * 0.85); x = _se_block(x)
        x = _residual_block(x, units_3, dropout_rate=dropout * 0.7);  x = _se_block(x)
        x = layers.Dense(32, activation="relu")(x); x = layers.Dropout(0.15)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = Model([num_input] + cat_inputs, out, name="DNN")

        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss="binary_crossentropy",
                      metrics=[keras.metrics.AUC(name="auc")])
        model.fit(X_train_inputs, y_train, validation_data=(X_val_inputs, y_val),
                  epochs=MAX_EPOCHS_HARD_CAP, batch_size=batch_size,
                  callbacks=[EarlyStopping(monitor="val_auc", patience=PATIENCE_EPOCHS,
                                           restore_best_weights=True, mode="max"),
                             WallClockBudget(MAX_WALLCLOCK_SECONDS_TRIAL)],
                  verbose=0)
        y_proba = model.predict(X_val_inputs, verbose=0).ravel()
        y_pred = (y_proba >= 0.5).astype(int)
        return f1_score(y_val, y_pred)
    return objective


def make_cnn_objective(input_dim, X_train, y_train, X_val, y_val):
    def objective(trial):
        filters_1 = trial.suggest_categorical("filters_1", [32, 64, 128])
        filters_2 = trial.suggest_categorical("filters_2", [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.15, 0.45, step=0.05)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

        inp = layers.Input(shape=(input_dim,))
        x = layers.Reshape((input_dim, 1))(inp)
        x = layers.Conv1D(filters_1, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters_2, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = Model(inp, out, name="CNN_1D")

        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss="binary_crossentropy",
                      metrics=[keras.metrics.AUC(name="auc")])
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=MAX_EPOCHS_HARD_CAP, batch_size=batch_size,
                  callbacks=[EarlyStopping(monitor="val_auc", patience=PATIENCE_EPOCHS,
                                           restore_best_weights=True, mode="max"),
                             WallClockBudget(MAX_WALLCLOCK_SECONDS_TRIAL)],
                  verbose=0)
        y_proba = model.predict(X_val, verbose=0).ravel()
        y_pred = (y_proba >= 0.5).astype(int)
        return f1_score(y_val, y_pred)
    return objective


def make_gru_objective(input_dim, X_train, y_train, X_val, y_val):
    def objective(trial):
        gru_units_1 = trial.suggest_categorical("gru_units_1", [32, 64, 128])
        gru_units_2 = trial.suggest_categorical("gru_units_2", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.15, 0.45, step=0.05)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

        inp = layers.Input(shape=(input_dim,))
        x = layers.Reshape((input_dim, 1))(inp)
        x = layers.GRU(gru_units_1, return_sequences=True)(x)
        x = layers.GRU(gru_units_2)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(dropout * 0.7)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = Model(inp, out, name="RNN_GRU")

        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss="binary_crossentropy",
                      metrics=[keras.metrics.AUC(name="auc")])
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=MAX_EPOCHS_HARD_CAP, batch_size=batch_size,
                  callbacks=[EarlyStopping(monitor="val_auc", patience=PATIENCE_EPOCHS,
                                           restore_best_weights=True, mode="max"),
                             WallClockBudget(MAX_WALLCLOCK_SECONDS_TRIAL)],
                  verbose=0)
        y_proba = model.predict(X_val, verbose=0).ravel()
        y_pred = (y_proba >= 0.5).astype(int)
        return f1_score(y_val, y_pred)
    return objective


def make_ae_objective(input_dim, X_train, y_train, X_val, y_val):
    def objective(trial):
        enc_1 = trial.suggest_categorical("enc_1", [64, 128, 256])
        enc_2 = trial.suggest_categorical("enc_2", [32, 64, 128])
        bottleneck = trial.suggest_categorical("bottleneck", [12, 24, 48])
        dropout = trial.suggest_float("dropout", 0.15, 0.40, step=0.05)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

        inp = layers.Input(shape=(input_dim,))
        enc = layers.Dense(enc_1, activation="relu")(inp)
        enc = layers.BatchNormalization()(enc)
        enc = layers.Dropout(dropout)(enc)
        enc = layers.Dense(enc_2, activation="relu")(enc)
        enc = layers.BatchNormalization()(enc)
        bn = layers.Dense(bottleneck, activation="relu", name="bottleneck")(enc)
        cls = layers.Dense(32, activation="relu")(bn)
        cls = layers.Dropout(dropout * 0.6)(cls)
        out = layers.Dense(1, activation="sigmoid", name="classification")(cls)
        model = Model(inp, out, name="AutoEncoder_Clf")

        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss="binary_crossentropy",
                      metrics=[keras.metrics.AUC(name="auc")])
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=MAX_EPOCHS_HARD_CAP, batch_size=batch_size,
                  callbacks=[EarlyStopping(monitor="val_auc", patience=PATIENCE_EPOCHS,
                                           restore_best_weights=True, mode="max"),
                             WallClockBudget(MAX_WALLCLOCK_SECONDS_TRIAL)],
                  verbose=0)
        y_proba = model.predict(X_val, verbose=0).ravel()
        y_pred = (y_proba >= 0.5).astype(int)
        return f1_score(y_val, y_pred)
    return objective


def make_xgb_objective(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        n_neg, n_pos = np.sum(y_train == 0), np.sum(y_train == 1)
        model = xgb.XGBClassifier(
            **params, scale_pos_weight=n_neg/n_pos,
            eval_metric="auc", random_state=SEED,
            use_label_encoder=False, verbosity=0, n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_proba = model.predict_proba(X_val)
        y_pred = model.predict(X_val)
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            y_proba = y_proba[:, 1]
        return f1_score(y_val, y_pred)
    return objective


OPTUNA_OBJECTIVES = {
    "DNN": make_dnn_objective,
    "CNN_1D": make_cnn_objective,
    "RNN_GRU": make_gru_objective,
    "AutoEncoder_Clf": make_ae_objective,
    "XGBoost": make_xgb_objective,
}


# ═══════════════════════════════════════════════════════════════════════
# 6. VISUALIZACIONES FASE 2
# ═══════════════════════════════════════════════════════════════════════

def plot_default_vs_optimized(default_results, optimized_results):
    """Bar chart comparativo Default vs Optimizado."""
    metric_keys = ["auc_roc", "auc_pr", "f1", "recall", "mcc"]
    metric_labels = ["AUC-ROC", "AUC-PR", "F1", "Recall", "MCC"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(MODEL_LABELS))
    width = 0.35

    for midx, mk in enumerate(metric_keys):
        fig_m, ax_m = plt.subplots(figsize=(10, 5))
        default_vals = [default_results[m]["metrics"][mk] for m in DEFAULT_BUILDERS]
        opt_vals = [optimized_results[m]["metrics"][mk] for m in DEFAULT_BUILDERS]

        bars1 = ax_m.bar(x - width/2, default_vals, width, label="Default",
                         color="#95a5a6", edgecolor="white")
        bars2 = ax_m.bar(x + width/2, opt_vals, width, label="Optimizado",
                         color="#27ae60", edgecolor="white")

        for bar, val in zip(bars1, default_vals):
            ax_m.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                      f"{val:.3f}", ha="center", fontsize=9, rotation=0)
        for bar, val in zip(bars2, opt_vals):
            ax_m.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                      f"{val:.3f}", ha="center", fontsize=9, rotation=0)

        ax_m.set_xticks(x)
        ax_m.set_xticklabels([MODEL_LABELS[m] for m in DEFAULT_BUILDERS])
        ax_m.set_title(f"Default vs Optimizado — {metric_labels[midx]}", fontweight="bold")
        ax_m.legend()
        ax_m.set_ylabel(metric_labels[midx])
        plt.tight_layout()
        fig_m.savefig(FIG_DIR / f"07_default_vs_opt_{mk}.png", bbox_inches="tight")
        plt.close(fig_m)

    print("  ✓ Gráficos Default vs Optimizado")


def generate_final_table(default_results, optimized_results):
    """Genera tabla final comparativa."""
    col_labels = ["Modelo", "Fase", "AUC-ROC", "AUC-PR", "F1",
                  "Recall", "Prec.", "MCC", "G-Mean", "Costo"]
    cell_data = []
    cell_colors = []

    best_val = 0
    for m in DEFAULT_BUILDERS:
        for res_dict, phase in [(default_results, "Default"), (optimized_results, "Optimizado")]:
            val = res_dict[m]["metrics"]["f1"]
            if val > best_val:
                best_val = val

    for m in DEFAULT_BUILDERS:
        for res_dict, phase in [(default_results, "Default"), (optimized_results, "Optimizado")]:
            met = res_dict[m]["metrics"]
            row = [
                MODEL_LABELS[m], phase,
                f"{met['auc_roc']:.4f}", f"{met['auc_pr']:.4f}",
                f"{met['f1']:.4f}", f"{met['recall']:.4f}",
                f"{met['precision']:.4f}", f"{met['mcc']:.4f}",
                f"{met['g_mean']:.4f}", f"{met['costo_negocio']:,}",
            ]
            cell_data.append(row)
            if met["f1"] == best_val:
                cell_colors.append(["#d5f5e3"] * len(row))
            else:
                cell_colors.append(["white"] * len(row))

    fig, ax = plt.subplots(figsize=(18, max(6, len(cell_data) * 0.4)))
    ax.axis("off")
    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     cellColours=cell_colors,
                     cellLoc="center", loc="center",
                     colColours=["#3498db"] * len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    for j in range(len(col_labels)):
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Fase 2: Default vs Optimizado — Resultados Finales",
                 fontweight="bold", fontsize=13, pad=20)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "08_tabla_final.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Tabla final")


def plot_roc_final(optimized_results, y_test):
    """Curvas ROC finales de los 5 modelos optimizados."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, res in optimized_results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, color=MODEL_COLORS.get(name, "gray"),
                label=f"{MODEL_LABELS[name]} (AUC={res['metrics']['auc_roc']:.4f})",
                linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.set_title("Curvas ROC — Modelos Optimizados", fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "09_roc_final.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Curvas ROC finales")


def plot_pr_final(optimized_results, y_test):
    """Curvas Precision-Recall finales."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, res in optimized_results.items():
        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        ax.plot(rec, prec, color=MODEL_COLORS.get(name, "gray"),
                label=f"{MODEL_LABELS[name]} (AP={res['metrics']['auc_pr']:.4f})",
                linewidth=2)
    baseline = y_test.mean()
    ax.axhline(baseline, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    ax.set_title("Curvas Precision-Recall — Modelos Optimizados", fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "10_pr_final.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Curvas PR finales")


def plot_confusion_final(optimized_results, y_test):
    """Matrices de confusión finales."""
    n = len(optimized_results)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, optimized_results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legítima", "Fraude"],
                    yticklabels=["Legítima", "Fraude"],
                    annot_kws={"size": 12})
        ax.set_title(MODEL_LABELS.get(name, name), fontweight="bold", fontsize=11)
        ax.set_ylabel("Real")
        ax.set_xlabel("Predicción")
    plt.suptitle("Matrices de Confusión — Modelos Optimizados",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "11_confusion_final.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Matrices de confusión finales")


# ═══════════════════════════════════════════════════════════════════════
# 7. SHAP
# ═══════════════════════════════════════════════════════════════════════

def run_shap(model, model_name, X_test, feature_names, is_xgb=False):
    """SHAP analysis para el mejor modelo."""
    print(f"\n[SHAP] Analizando {MODEL_LABELS[model_name]}...")
    import shap

    bg_size = min(200, len(X_test))
    bg_indices = np.random.choice(len(X_test), bg_size, replace=False)
    background = X_test[bg_indices]
    np.save(MODEL_DIR / "background.npy", background)

    try:
        if is_xgb:
            explainer = shap.TreeExplainer(model)
            test_sample = X_test[:500]
            shap_values = explainer.shap_values(test_sample)
        else:
            explainer = shap.DeepExplainer(model, background)
            test_sample = X_test[:500]
            shap_values = explainer.shap_values(test_sample)
    except Exception:
        print("  Usando KernelExplainer (fallback)...")
        if is_xgb:
            predict_fn = lambda x: model.predict_proba(x)[:, 1]
        else:
            predict_fn = lambda x: model.predict(x, verbose=0).ravel()
        explainer = shap.KernelExplainer(predict_fn, background)
        test_sample = X_test[:200]
        shap_values = explainer.shap_values(test_sample, nsamples=100)

    if isinstance(shap_values, list):
        sv = shap_values[0]
    else:
        sv = shap_values
    sv = np.squeeze(sv)

    short_names = []
    for fn in feature_names:
        fn = fn.replace("num__", "").replace("cat__", "")
        if len(fn) > 25:
            fn = fn[:22] + "..."
        short_names.append(fn)

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    mean_abs = np.abs(sv).mean(axis=0)
    top_k = 20
    top_idx = np.argsort(mean_abs)[-top_k:]
    sorted_idx = top_idx[np.argsort(mean_abs[top_idx])]

    ax.barh(range(top_k), mean_abs[sorted_idx],
            color=sns.color_palette("YlOrRd", top_k), edgecolor="white")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([short_names[i] for i in sorted_idx], fontsize=9)
    ax.set_xlabel("Importancia SHAP Promedio (|SHAP value|)")
    ax.set_title(f"Importancia de Features — {MODEL_LABELS[model_name]} (SHAP)",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(XAI_DIR / "01_shap_importancia.png", bbox_inches="tight")
    plt.close()

    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv[:, sorted_idx], test_sample[:len(sv), sorted_idx],
                         feature_names=[short_names[i] for i in sorted_idx],
                         show=False, max_display=20)
        plt.title(f"SHAP Summary — {MODEL_LABELS[model_name]}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(XAI_DIR / "02_shap_summary.png", bbox_inches="tight", dpi=300)
        plt.close("all")
    except Exception as e:
        print(f"  ⚠ SHAP Summary falló: {e}")

    # Group map
    group_map = {}
    for i, fn in enumerate(feature_names):
        base = fn.split("_")[0] if "_" in fn else fn
        if base not in group_map:
            group_map[base] = []
        group_map[base].append(fn)
    with open(MODEL_DIR / "group_map.json", "w") as f:
        json.dump(group_map, f, indent=2)

    print("  ✓ SHAP completado")
    return sv


def run_shap_dnn_emb(model, dnn_test_inputs, embed_cols, dnn_data, n_samples=150):
    """SHAP for DAFD-Net with embeddings using KernelExplainer.

    Concatenates the dense stream and ordinal categoricals into a flat
    matrix, then defines a predict-fn that splits and feeds the multi-input
    model. KernelExplainer is slower than DeepExplainer but supports any
    multi-input model.
    """
    print("\n[SHAP] Analizando DAFD-Net (Embeddings) via KernelExplainer...")
    import shap

    test_dense = dnn_test_inputs[0].astype(np.float32)
    test_ords  = np.hstack(dnn_test_inputs[1:]).astype(np.float32)
    flat_test  = np.hstack([test_dense, test_ords]).astype(np.float32)
    dense_dim  = test_dense.shape[1]
    n_ord_cols = test_ords.shape[1]

    def predict_fn(X_flat):
        dense_part = X_flat[:, :dense_dim].astype(np.float32)
        ord_parts  = [X_flat[:, dense_dim + j:dense_dim + j + 1].astype(np.int32)
                       for j in range(n_ord_cols)]
        return model.predict([dense_part] + ord_parts, verbose=0).ravel()

    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(flat_test.shape[0], min(50, flat_test.shape[0]), replace=False)
    background = flat_test[bg_idx]
    np.save(MODEL_DIR / "background.npy", background)

    sample_idx = rng.choice(flat_test.shape[0],
                             min(n_samples, flat_test.shape[0]), replace=False)
    sample = flat_test[sample_idx]

    explainer = shap.KernelExplainer(predict_fn, background)
    sv = explainer.shap_values(sample, nsamples=50)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.squeeze(sv)

    short_names = [f"dense_{i}" for i in range(dense_dim)] + list(embed_cols)
    fig, ax = plt.subplots(figsize=(10, 8))
    mean_abs = np.abs(sv).mean(axis=0)
    top_k = min(20, len(mean_abs))
    top_idx = np.argsort(mean_abs)[-top_k:]
    sorted_idx = top_idx[np.argsort(mean_abs[top_idx])]
    ax.barh(range(top_k), mean_abs[sorted_idx],
            color=sns.color_palette("YlOrRd", top_k), edgecolor="white")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([short_names[i] if i < len(short_names) else f"f_{i}"
                         for i in sorted_idx], fontsize=9)
    ax.set_xlabel("Importancia SHAP Promedio (|SHAP value|)")
    ax.set_title("Importancia SHAP — DAFD-Net con Embeddings (KernelExplainer)",
                  fontweight="bold")
    plt.tight_layout()
    fig.savefig(XAI_DIR / "01_shap_importancia.png", bbox_inches="tight")
    plt.close()

    # Group map for embeddings
    group_map = {"dense_stream": [f"dense_{i}" for i in range(dense_dim)],
                  "embeddings": list(embed_cols)}
    with open(MODEL_DIR / "group_map.json", "w") as f:
        json.dump(group_map, f, indent=2)

    print(f"  ✓ SHAP DNN-Emb: {sample.shape[0]} samples × {flat_test.shape[1]} features")
    return sv


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("FASE 2: OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("Default vs Optimizado (Optuna)")
    print("=" * 70)

    # Load Phase 1 results
    phase1_path = MODEL_DIR / "balance_experiment_results.json"
    if not phase1_path.exists():
        print("⚠ balance_experiment_results.json no encontrado.")
        print("  Ejecuta primero: python 03_balance_experiment.py")
        return

    with open(phase1_path) as f:
        phase1 = json.load(f)

    best_balance = phase1["best_balance"]
    # NEW: per-model best balance — each model trains with its own optimal balancing
    best_balance_per_model = phase1.get("best_balance_per_model", {
        m: best_balance for m in DEFAULT_BUILDERS
    })
    print(f"\n  Mejor balanceo global de Fase 1: {BALANCE_LABELS[best_balance]}")
    print(f"  Per-model best balance:")
    for m, b in best_balance_per_model.items():
        print(f"    {MODEL_LABELS.get(m, m):14s} → {BALANCE_LABELS.get(b, b)}")

    # Load & prepare data (returns balanced variants for ALL 3 methods)
    (X_train, y_train, X_val, y_val,
     X_test, y_test, feature_names, preprocessor, dnn_data,
     balanced_ohe) = load_and_prepare(best_balance)
    input_dim = X_train.shape[1]

    # DNN val/test inputs are unbalanced (we only balance the training set)
    def _dnn_inputs(dense_arr, ord_arr):
        return [dense_arr] + [ord_arr[:, j:j+1] for j in range(ord_arr.shape[1])]
    dnn_val_inputs   = _dnn_inputs(dnn_data["X_val_dense"],   dnn_data["X_val_ord"])
    dnn_test_inputs  = _dnn_inputs(dnn_data["X_test_dense"],  dnn_data["X_test_ord"])
    embed_dims_default = list(EMBED_CATEGORICAL_FEATURES.values())

    def _get_train_data_for_model(model_name):
        """Returns the (X_train, y_train) balanced for this model's preferred method."""
        bal = best_balance_per_model.get(model_name, best_balance)
        if model_name == "DNN":
            Xd, Xo, yb = dnn_data["balanced_dnn"][bal]
            return _dnn_inputs(Xd, Xo), yb, bal
        else:
            Xb, yb = balanced_ohe[bal]
            return Xb, yb, bal

    # ─── FASE DEFAULT ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[A] ENTRENAMIENTO CON HIPERPARÁMETROS DEFAULT (per-model balance)")
    print("=" * 70)

    default_results = OrderedDict()
    for name, builder in DEFAULT_BUILDERS.items():
        is_xgb = (name == "XGBoost")
        is_dnn = (name == "DNN")
        Xtr, ytr, bal_used = _get_train_data_for_model(name)
        print(f"\n  {MODEL_LABELS[name]} [balance={BALANCE_LABELS.get(bal_used, bal_used)}]")

        if is_xgb:
            model = builder(input_dim)
            model, hist, t_time = train_xgb(model, Xtr, ytr, X_val, y_val, name)
            metrics, y_proba, y_pred = evaluate(model, X_test, y_test, is_xgb=True)
        elif is_dnn:
            model = build_dnn_default(
                dnn_data["dense_dim"],
                cardinalities=dnn_data["cardinalities"],
                embed_dims=embed_dims_default,
                embed_cols=EMBED_COLS,
            )
            model, hist, t_time = train_keras(
                model, Xtr, ytr, dnn_val_inputs, y_val, name
            )
            metrics, y_proba, y_pred = evaluate(model, dnn_test_inputs, y_test, is_xgb=False)
        else:
            model = builder(input_dim)
            model, hist, t_time = train_keras(model, Xtr, ytr, X_val, y_val, name)
            metrics, y_proba, y_pred = evaluate(model, X_test, y_test, is_xgb=False)
        default_results[name] = {
            "model": model, "metrics": metrics,
            "y_proba": y_proba, "y_pred": y_pred,
            "train_time": t_time, "balance_used": bal_used,
        }
        print(f"    AUC-ROC={metrics['auc_roc']:.4f} | Recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f}")
        gc.collect()

    # ─── FASE OPTUNA ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"[B] OPTIMIZACIÓN CON OPTUNA (trials por modelo: {OPTUNA_TRIALS_PER_MODEL})")
    print("=" * 70)

    optimized_results = OrderedDict()
    for name in DEFAULT_BUILDERS:
        n_trials = OPTUNA_TRIALS_PER_MODEL.get(name, OPTUNA_TRIALS)
        is_xgb = (name == "XGBoost")
        is_dnn = (name == "DNN")

        # Skip Optuna for models with 0 trials — copy default result directly
        if n_trials == 0:
            print(f"\n  {MODEL_LABELS[name]}: Optuna saltado (0 trials) — usando defaults")
            def_r = default_results[name]
            optimized_results[name] = {
                "model": def_r["model"], "metrics": def_r["metrics"],
                "y_proba": def_r["y_proba"], "y_pred": def_r["y_pred"],
                "train_time": def_r["train_time"],
                "best_params": {},
                "balance_used": def_r.get("balance_used", best_balance),
            }
            print(f"    (default) AUC-ROC={def_r['metrics']['auc_roc']:.4f} | "
                  f"Recall={def_r['metrics']['recall']:.4f} | MCC={def_r['metrics']['mcc']:.4f}")
            gc.collect()
            continue

        # Per-model best balance lookup
        Xtr_opt, ytr_opt, bal_used_opt = _get_train_data_for_model(name)
        print(f"\n  Optimizando {MODEL_LABELS[name]} ({n_trials} trials) "
              f"[balance={BALANCE_LABELS.get(bal_used_opt, bal_used_opt)}]...")

        if is_xgb:
            obj = OPTUNA_OBJECTIVES[name](Xtr_opt, ytr_opt, X_val, y_val)
        elif is_dnn:
            obj = make_dnn_objective(
                dnn_data["dense_dim"],
                Xtr_opt, ytr_opt,
                dnn_val_inputs, y_val,
                dnn_data["cardinalities"], EMBED_COLS,
            )
        else:
            obj = OPTUNA_OBJECTIVES[name](input_dim, Xtr_opt, ytr_opt, X_val, y_val)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

        print(f"    Mejor trial: F1={study.best_value:.4f}")
        print(f"    Params: {study.best_params}")

        # Retrain with best params
        bp = study.best_params
        if is_xgb:
            n_neg, n_pos = np.sum(ytr_opt == 0), np.sum(ytr_opt == 1)
            model = xgb.XGBClassifier(
                n_estimators=bp.get("n_estimators", 300),
                max_depth=bp.get("max_depth", 6),
                learning_rate=bp.get("lr", 0.1),
                subsample=bp.get("subsample", 0.8),
                colsample_bytree=bp.get("colsample", 0.8),
                reg_alpha=bp.get("reg_alpha", 0.1),
                reg_lambda=bp.get("reg_lambda", 1.0),
                min_child_weight=bp.get("min_child", 1),
                gamma=bp.get("gamma", 0.0),
                scale_pos_weight=n_neg/n_pos,
                eval_metric="auc", random_state=SEED,
                use_label_encoder=False, verbosity=0, n_jobs=-1,
            )
            model, _, t_time = train_xgb(model, Xtr_opt, ytr_opt, X_val, y_val, name)
            metrics, y_proba, y_pred = evaluate(model, X_test, y_test, is_xgb=True)
        elif is_dnn:
            # Retrain DNN-Emb with Optuna best params + embedding scaling
            emb_scale = bp.get("emb_scale", 1.0)
            best_dims = [max(2, int(round(d * emb_scale))) for d in embed_dims_default]
            num_input = layers.Input(shape=(dnn_data["dense_dim"],), name="dense_in")
            cat_inputs, cat_embeds = [], []
            for col, card, dim in zip(EMBED_COLS, dnn_data["cardinalities"], best_dims):
                ci = layers.Input(shape=(1,), name=f"cat_{col}", dtype="int32")
                ce = layers.Embedding(card, dim, name=f"emb_{col}")(ci)
                ce = layers.Flatten(name=f"flat_{col}")(ce)
                cat_inputs.append(ci); cat_embeds.append(ce)
            xf = layers.Concatenate(name="fusion")([num_input] + cat_embeds)
            xf = layers.Dense(bp.get("units_1", 384), activation="relu")(xf)
            xf = layers.BatchNormalization()(xf)
            xf = layers.Dropout(bp.get("dropout", 0.3))(xf)
            xf = _residual_block(xf, bp.get("units_2", 192),
                                  dropout_rate=bp.get("dropout", 0.3) * 0.85); xf = _se_block(xf)
            xf = _residual_block(xf, bp.get("units_3", 96),
                                  dropout_rate=bp.get("dropout", 0.3) * 0.7);  xf = _se_block(xf)
            xf = layers.Dense(32, activation="relu")(xf); xf = layers.Dropout(0.15)(xf)
            out = layers.Dense(1, activation="sigmoid")(xf)
            model = Model([num_input] + cat_inputs, out, name="DNN")
            model, _, t_time = train_keras(
                model, Xtr_opt, ytr_opt,
                dnn_val_inputs, y_val, name,
                lr=bp.get("lr", 0.001),
                batch_size=bp.get("batch_size", 256),
            )
            metrics, y_proba, y_pred = evaluate(model, dnn_test_inputs, y_test, is_xgb=False)
        else:
            # Rebuild non-DNN model with Optuna params
            lr = bp.get("lr", 0.001)
            batch_size = bp.get("batch_size", 256)
            model = DEFAULT_BUILDERS[name](input_dim)
            model, _, t_time = train_keras(
                model, Xtr_opt, ytr_opt, X_val, y_val, name,
                lr=lr, batch_size=batch_size
            )
            metrics, y_proba, y_pred = evaluate(model, X_test, y_test, is_xgb=False)
        optimized_results[name] = {
            "model": model, "metrics": metrics,
            "y_proba": y_proba, "y_pred": y_pred,
            "train_time": t_time,
            "best_params": bp,
            "balance_used": bal_used_opt,
        }
        print(f"    Final: AUC-ROC={metrics['auc_roc']:.4f} | "
              f"Recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f} | MCC={metrics['mcc']:.4f}")
        gc.collect()

    # ─── COMPARATIVA ──────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("COMPARATIVA DEFAULT vs OPTIMIZADO")
    print("=" * 100)
    header = f"{'Modelo':<14} {'Fase':<12} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>8} {'Recall':>8} {'MCC':>8} {'Costo':>8}"
    print(header)
    print("-" * 100)
    for name in DEFAULT_BUILDERS:
        for res_dict, phase in [(default_results, "Default"), (optimized_results, "Optimizado")]:
            m = res_dict[name]["metrics"]
            print(f"{MODEL_LABELS[name]:<14} {phase:<12} {m['auc_roc']:>8.4f} "
                  f"{m['auc_pr']:>8.4f} {m['f1']:>8.4f} {m['recall']:>8.4f} "
                  f"{m['mcc']:>8.4f} {m['costo_negocio']:>8,}")
        print()

    # Determine overall winner
    # Determine overall winner by F1
    best_f1 = 0
    overall_winner = ""
    for name in DEFAULT_BUILDERS:
        f1 = optimized_results[name]["metrics"]["f1"]
        if f1 > best_f1:
            best_f1 = f1
            overall_winner = name

    print(f"🏆 GANADOR GLOBAL: {MODEL_LABELS[overall_winner]} (F1-Score={best_f1:.4f})")
    # Determine DL winner (for export)
    dl_models = [n for n in DEFAULT_BUILDERS if n != "XGBoost"]
    best_dl_f1 = 0
    dl_winner = ""
    for name in dl_models:
        f1 = optimized_results[name]["metrics"]["f1"]
        if f1 > best_dl_f1:
            best_dl_f1 = f1
            dl_winner = name
    
    if dl_winner != overall_winner:
        print(f"  ℹ️  {MODEL_LABELS[overall_winner]} ganó analíticamente pero NO se exporta al backend.")

    export_model_name = dl_winner
    export_model = optimized_results[dl_winner]["model"]

    if overall_winner == "XGBoost":
        print(f"  ℹ️  XGBoost ganó analíticamente pero NO se exporta al backend.")
        print(f"  📦 Se exporta: {MODEL_LABELS[dl_winner]} (mejor DL, F1={best_dl_f1:.4f})")
    else:
        export_model_name = overall_winner
        export_model = optimized_results[overall_winner]["model"]

    # ─── VISUALIZACIONES ─────────────────────────────────────────
    print("\n[VISUALIZACIONES]")
    print("-" * 50)
    plot_default_vs_optimized(default_results, optimized_results)
    generate_final_table(default_results, optimized_results)
    plot_roc_final(optimized_results, y_test)
    plot_pr_final(optimized_results, y_test)
    plot_confusion_final(optimized_results, y_test)

    # ─── SHAP ────────────────────────────────────────────────────
    is_xgb_export = (export_model_name == "XGBoost")
    is_dnn_emb_export = (export_model_name == "DNN" and len(export_model.inputs) > 1)
    if is_dnn_emb_export:
        try:
            run_shap_dnn_emb(export_model, dnn_test_inputs, EMBED_COLS, dnn_data)
        except Exception as e:
            print(f"  ⚠ SHAP DNN-Emb falló: {e}")
    else:
        try:
            run_shap(export_model, export_model_name, X_test,
                     feature_names, is_xgb=is_xgb_export)
        except Exception as e:
            print(f"  ⚠ SHAP falló: {e}")

    # ─── GUARDAR ─────────────────────────────────────────────────
    print("\n[GUARDANDO MODELO FINAL]")
    print("-" * 50)
    model_path = MODEL_DIR / "best_model.keras"
    export_model.save(model_path)
    print(f"  ✓ Modelo exportado: {model_path}")
    print(f"  ✓ Modelo: {MODEL_LABELS[export_model_name]}")

    # Save all optimized models
    for name, res in optimized_results.items():
        is_xgb = (name == "XGBoost")
        if is_xgb:
            res["model"].save_model(str(MODEL_DIR / f"{name}_optimized.json"))
        else:
            res["model"].save(MODEL_DIR / f"{name}_optimized.keras")

    # Save comparison metrics
    metrics_export = {
        "best_balance": best_balance,
        "overall_winner": overall_winner,
        "export_model": export_model_name,
        "default": {},
        "optimized": {},
    }
    for name in DEFAULT_BUILDERS:
        metrics_export["default"][name] = {
            "label": MODEL_LABELS[name],
            "metrics": default_results[name]["metrics"],
            "train_time": default_results[name]["train_time"],
        }
        opt_entry = {
            "label": MODEL_LABELS[name],
            "metrics": optimized_results[name]["metrics"],
            "train_time": optimized_results[name]["train_time"],
        }
        if "best_params" in optimized_results[name]:
            opt_entry["best_params"] = optimized_results[name]["best_params"]
        metrics_export["optimized"][name] = opt_entry

    with open(MODEL_DIR / "comparison_metrics.json", "w") as f:
        json.dump(metrics_export, f, indent=2, default=str)

    export_plot_data_04(default_results, optimized_results, y_test)

    # ─── RESUMEN ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("✅ FASE 2 COMPLETADA")
    print(f"  📊 Gráficos: {FIG_DIR}")
    print(f"  🧠 Modelos: {MODEL_DIR}")
    print(f"  🔍 SHAP: {XAI_DIR}")
    bm = optimized_results[export_model_name]["metrics"]
    print(f"\n  🏆 Ganador global:  {MODEL_LABELS[overall_winner]}")
    print(f"  📦 Modelo export.:  {MODEL_LABELS[export_model_name]}")
    print(f"     AUC-ROC:    {bm['auc_roc']:.4f}")
    print(f"     AUC-PR:     {bm['auc_pr']:.4f}")
    print(f"     F1-Score:   {bm['f1']:.4f}")
    print(f"     Recall:     {bm['recall']:.4f}")
    print(f"     MCC:        {bm['mcc']:.4f}")
    print("\n  ➡️  Ejecuta '05_export_model.py' para copiar al backend")
    print("=" * 70)


def export_plot_data_04(default_results, optimized_results, y_test):
    """Extrae la data para regenerar los gráficos de la Fase 2 en JSON."""
    import json
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
    print("\n  Generando 04_plot_data.json...")
    
    plot_data = {
        "roc_curves": {},
        "pr_curves": {},
        "confusion_matrices": {}
    }
    
    for name, res in optimized_results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        step = max(1, len(fpr) // 200)
        plot_data["roc_curves"][name] = {
            "fpr": fpr[::step].tolist(),
            "tpr": tpr[::step].tolist()
        }
        
        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        step_pr = max(1, len(prec) // 200)
        plot_data["pr_curves"][name] = {
            "precision": prec[::step_pr].tolist(),
            "recall": rec[::step_pr].tolist()
        }
        
        cm = confusion_matrix(y_test, res["y_pred"])
        plot_data["confusion_matrices"][name] = cm.tolist()
        
    out_path = FIG_DIR / "04_plot_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plot_data, f, indent=2, ensure_ascii=False)
    print("  ✓ Datos JSON exportados a 04_plot_data.json")

if __name__ == "__main__":
    main()
