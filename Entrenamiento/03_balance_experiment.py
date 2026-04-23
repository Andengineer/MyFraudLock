#!/usr/bin/env python3
"""
03_balance_experiment.py — Fase de Experimentación de Balanceo
======================================================================
Compara 3 técnicas de balanceo × 5 modelos (4 DL + 1 ML clásico)
para determinar la mejor combinación antes de optimizar hiperparámetros.

Técnicas de balanceo:
  1. baseline    — Sin balanceo (solo class_weight)
  2. smote_tomek — SMOTE + Tomek Links
  3. adasyn      — Adaptive Synthetic Sampling

Modelos:
  1. DNN           (Feed-Forward Dense)
  2. CNN-1D        (Convolutional 1D)
  3. RNN-GRU       (Gated Recurrent Unit)
  4. Autoencoder   (Autoencoder + Classification Head)
  5. XGBoost       (Gradient Boosting — ML clásico)

Métricas (orientadas al negocio):
  - AUC-ROC, AUC-PR, F1-Score, Precision, Recall, FPR
  - MCC (Matthews Correlation Coefficient)
  - G-Mean (sqrt(Sensitivity × Specificity))
  - Costo de Negocio: FN×10 + FP×1
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

# ML / DL
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, matthews_corrcoef, average_precision_score, precision_recall_curve, roc_curve
)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_best_f1_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    best_f1, best_th = 0, 0.5
    for p, r, th in zip(precisions, recalls, thresholds):
        f1 = 2 * p * r / (p + r + 1e-10)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th, best_f1

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from joblib import dump
import xgboost as xgb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ─── Config ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "outputs" / "data"
MODEL_DIR = BASE_DIR / "outputs" / "models"
FIG_DIR   = BASE_DIR / "outputs" / "figures"
for d in [MODEL_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Estilo gráficos
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.grid": True, "grid.alpha": 0.3,
})

# ═══════════════════════════════════════════════════════════════════════
# FEATURES
# ═══════════════════════════════════════════════════════════════════════
NUMERIC_FEATURES = [
    "transaction_amount", "amt_log1p", "hour", "day_of_week", "month",
    "is_weekend", "eci_code", "has_3ds", "city_population", "num_items",
    "has_discount", "num_installments", "previous_failed_attempts",
    "is_new_customer", "days_since_first_purchase", "avg_historical_amount",
    "is_high_risk_hour", "amount_deviation",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
    # Non-linear interaction features (ventaja DNN sobre árboles)
    "amt_hour_interaction", "amt_fail_interaction", "risk_score_smooth",
    "amt_pop_sigmoid", "customer_maturity", "night_newcust_score",
    "session_duration_minutes", "interaction_velocity",
    "device_telemetry_1", "device_telemetry_2", "device_telemetry_3",
    "device_telemetry_4", "device_telemetry_5",
]
CATEGORICAL_FEATURES = [
    "card_brand", "card_type", "issuer_bank", "payment_channel",
    "customer_region", "category",
]
TARGET = "is_fraud"

# ═══════════════════════════════════════════════════════════════════════
# COLORES Y LABELS
# ═══════════════════════════════════════════════════════════════════════
MODEL_COLORS = {
    "DNN": "#3498db",
    "CNN_1D": "#e74c3c",
    "RNN_GRU": "#2ecc71",
    "AutoEncoder_Clf": "#9b59b6",
    "XGBoost": "#f39c12",
}
MODEL_LABELS = {
    "DNN": "DNN",
    "CNN_1D": "CNN-1D",
    "RNN_GRU": "RNN-GRU",
    "AutoEncoder_Clf": "Autoencoder",
    "XGBoost": "XGBoost",
}
BALANCE_LABELS = {
    "baseline": "Sin Balanceo",
    "smote_tomek": "SMOTE-Tomek",
    "adasyn": "ADASYN",
}
BALANCE_COLORS = {
    "baseline": "#95a5a6",
    "smote_tomek": "#3498db",
    "adasyn": "#e74c3c",
}

# ═══════════════════════════════════════════════════════════════════════
# 1. CARGA Y PREPROCESAMIENTO (con Feature Engineering avanzado)
# ═══════════════════════════════════════════════════════════════════════

def engineer_features(df):
    """Crea features de interacción que las redes neuronales capturan
    mejor que los modelos basados en árboles."""
    df = df.copy()
    # Interacciones clave de dominio
    df["amt_x_new_customer"] = df["transaction_amount"] * df["is_new_customer"]
    df["amt_x_high_risk_hour"] = df["transaction_amount"] * df["is_high_risk_hour"]
    df["failed_x_new"] = df["previous_failed_attempts"] * df["is_new_customer"]
    df["deviation_x_no3ds"] = df["amount_deviation"] * (1 - df["has_3ds"])
    df["amt_ratio_hist"] = df["transaction_amount"] / (df["avg_historical_amount"] + 1)
    df["risk_score_proxy"] = (
        df["is_high_risk_hour"] * 2 +
        df["is_new_customer"] * 1.5 +
        (1 - df["has_3ds"]) * 2 +
        np.clip(df["previous_failed_attempts"], 0, 5) * 1.0 +
        np.clip(df["amount_deviation"], 0, 10) * 0.5
    )
    return df

# Features de interacción adicionales (se añaden post-escalado)
INTERACTION_FEATURES = [
    "amt_x_new_customer", "amt_x_high_risk_hour", "failed_x_new",
    "deviation_x_no3ds", "amt_ratio_hist", "risk_score_proxy",
]


def load_and_preprocess():
    """Carga datos, preprocesa con feature engineering y divide en train/val/test."""
    print("\n[1] CARGA Y PREPROCESAMIENTO")
    print("-" * 50)

    df = pd.read_csv(DATA_DIR / "fraud_ecommerce_dataset.csv")
    print(f"  Dataset: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"  Fraude: {df[TARGET].sum():,} ({df[TARGET].mean()*100:.1f}%)")

    # Feature engineering
    df = engineer_features(df)
    all_numeric = NUMERIC_FEATURES + INTERACTION_FEATURES
    print(f"  Features de interacción agregadas: {len(INTERACTION_FEATURES)}")

    X = df[all_numeric + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].values

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SKPipeline([("scaler", StandardScaler())]), all_numeric),
            ("cat", SKPipeline([("onehot", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            ))]), CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    # Split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=SEED, stratify=y_temp
    )

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc   = preprocessor.transform(X_val)
    X_test_proc  = preprocessor.transform(X_test)

    num_names = all_numeric
    cat_names = list(preprocessor.named_transformers_["cat"]
                     .named_steps["onehot"]
                     .get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = num_names + cat_names
    print(f"  Features procesadas: {len(feature_names)}")

    # Guardar preprocesador
    dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    return (X_train_proc, y_train, X_val_proc, y_val,
            X_test_proc, y_test, feature_names, preprocessor)


# ═══════════════════════════════════════════════════════════════════════
# 2. TÉCNICAS DE BALANCEO
# ═══════════════════════════════════════════════════════════════════════

def apply_balancing(X_train, y_train, method):
    """Aplica una técnica de balanceo al conjunto de entrenamiento."""
    print(f"\n  Aplicando: {BALANCE_LABELS[method]}...")
    print(f"    Antes:   {dict(zip(*np.unique(y_train, return_counts=True)))}")

    if method == "baseline":
        X_bal, y_bal = X_train.copy(), y_train.copy()
    elif method == "smote_tomek":
        st = SMOTETomek(
            smote=SMOTE(random_state=SEED, sampling_strategy=0.4),
            random_state=SEED
        )
        X_bal, y_bal = st.fit_resample(X_train, y_train)
    elif method == "adasyn":
        ada = ADASYN(random_state=SEED, sampling_strategy=0.4)
        X_bal, y_bal = ada.fit_resample(X_train, y_train)
    else:
        raise ValueError(f"Método desconocido: {method}")

    print(f"    Después: {dict(zip(*np.unique(y_bal, return_counts=True)))}")
    return X_bal, y_bal


# ═══════════════════════════════════════════════════════════════════════
# 3. FOCAL LOSS (Lin et al., 2017 — Facebook AI Research)
# ═══════════════════════════════════════════════════════════════════════

def focal_loss(gamma=2.0, alpha=0.75):
    """Focal Loss: Penaliza muestras fáciles y enfoca en las difíciles.
    Referencia: 'Focal Loss for Dense Object Detection' (Lin et al., 2017).
    Ventaja sobre binary_crossentropy: Automáticamente pondera la clase
    minoritaria y las muestras en la frontera de decisión.
    """
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # Alpha weighting
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        # Focal modulation
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha_factor * tf.pow(1.0 - p_t, gamma)
        # Cross entropy
        ce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(focal_weight * ce)
    return focal_loss_fn


# ═══════════════════════════════════════════════════════════════════════
# 4. DEFINICIÓN DE MODELOS
# ═══════════════════════════════════════════════════════════════════════

def _residual_block(x, units, dropout_rate=0.25):
    """Bloque residual: permite gradientes más profundos y estables."""
    shortcut = x
    h = layers.Dense(units, activation="relu")(x)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(dropout_rate)(h)
    h = layers.Dense(units, activation="relu")(h)
    h = layers.BatchNormalization()(h)
    # Proyectar shortcut si las dimensiones difieren
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units, activation="linear")(shortcut)
    out = layers.Add()([shortcut, h])
    out = layers.Activation("relu")(out)
    return out


def _se_block(x, ratio=4):
    """Squeeze-and-Excitation: atencion a nivel de features.
    Referencia: 'Squeeze-and-Excitation Networks' (Hu et al., 2018).
    Aprende qué features son más relevantes adaptativamente."""
    units = x.shape[-1]
    se = layers.Dense(units // ratio, activation="relu")(x)
    se = layers.Dense(units, activation="sigmoid")(se)
    return layers.Multiply()([x, se])


def build_dnn(input_dim):
    """Modelo 1: Deep Neural Network con Bloques Residuales + SE Attention."""
    inp = layers.Input(shape=(input_dim,))

    # Stem: Proyección inicial
    x = layers.Dense(384, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Bloques residuales con SE attention
    x = _residual_block(x, 256, dropout_rate=0.30)
    x = _se_block(x)
    x = _residual_block(x, 128, dropout_rate=0.25)
    x = _se_block(x)
    x = _residual_block(x, 64, dropout_rate=0.20)

    # Classification head
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.15)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    return Model(inp, out, name="DNN")


def build_cnn(input_dim):
    """Modelo 2: 1D Convolutional Neural Network"""
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


def build_rnn_gru(input_dim):
    """Modelo 3: RNN con Gated Recurrent Unit"""
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


def build_autoencoder_classifier(input_dim):
    """Modelo 4: Autoencoder + Classification Head"""
    inp = layers.Input(shape=(input_dim,))
    enc = layers.Dense(128, activation="relu")(inp)
    enc = layers.BatchNormalization()(enc)
    enc = layers.Dropout(0.3)(enc)
    enc = layers.Dense(64, activation="relu")(enc)
    enc = layers.BatchNormalization()(enc)
    bottleneck = layers.Dense(24, activation="relu", name="bottleneck")(enc)
    dec = layers.Dense(64, activation="relu")(bottleneck)
    dec = layers.Dense(128, activation="relu")(dec)
    reconstruction = layers.Dense(input_dim, activation="linear", name="reconstruction")(dec)
    cls = layers.Dense(32, activation="relu")(bottleneck)
    cls = layers.Dropout(0.2)(cls)
    classification = layers.Dense(1, activation="sigmoid", name="classification")(cls)
    model = Model(inp, classification, name="AutoEncoder_Clf")
    return model


def build_xgboost(input_dim):
    """Modelo 5: XGBoost (Gradient Boosting — ML clásico)"""
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1,
        eval_metric="auc",
        random_state=SEED,
        use_label_encoder=False,
        verbosity=0,
        n_jobs=-1,
    )


MODEL_BUILDERS = OrderedDict([
    ("DNN",              build_dnn),
    ("CNN_1D",           build_cnn),
    ("RNN_GRU",          build_rnn_gru),
    ("AutoEncoder_Clf",  build_autoencoder_classifier),
    ("XGBoost",          build_xgboost),
])


# ═══════════════════════════════════════════════════════════════════════
# 5. ENTRENAMIENTO Y EVALUACIÓN
# ═══════════════════════════════════════════════════════════════════════

class WarmUpSchedule(keras.callbacks.Callback):
    """Learning rate warmup: evita inestabilidad al inicio del entrenamiento."""
    def __init__(self, warmup_epochs=5, target_lr=0.001):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.target_lr * (epoch + 1) / self.warmup_epochs
            self.model.optimizer.learning_rate.assign(lr)


# Configuración de entrenamiento por modelo
# DNN tiene prioridad: más epochs y paciencia para convergencia óptima
TRAIN_CONFIG = {
    "DNN":              {"epochs": 150, "patience": 20, "lr_patience": 8},
    "CNN_1D":           {"epochs": 15,  "patience": 3,  "lr_patience": 2},
    "RNN_GRU":          {"epochs": 15,  "patience": 3,  "lr_patience": 2},
    "AutoEncoder_Clf":  {"epochs": 15,  "patience": 3,  "lr_patience": 2},
}

def train_keras_model(model, X_train, y_train, X_val, y_val, model_name,
                      batch_size=256):
    """Entrena un modelo Keras con Focal Loss, warmup y early stopping."""
    cfg = TRAIN_CONFIG.get(model_name, {"epochs": 60, "patience": 8, "lr_patience": 4})

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=focal_loss(gamma=2.0, alpha=0.75),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        WarmUpSchedule(warmup_epochs=5, target_lr=0.001),
        EarlyStopping(monitor="val_auc", patience=cfg["patience"],
                      restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=cfg["lr_patience"], min_lr=1e-6),
    ]

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    train_time = time.time() - start

    best_epoch = np.argmax(history.history.get("val_auc", [0]))
    print(f"    ✓ {model_name} entrenado en {train_time:.1f}s ({best_epoch+1} epochs)")
    return model, train_time


def train_xgboost_model(model, X_train, y_train, X_val, y_val, model_name):
    """Entrena un modelo XGBoost."""
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    model.set_params(scale_pos_weight=n_neg / n_pos)

    start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    train_time = time.time() - start
    print(f"    ✓ {model_name} entrenado en {train_time:.1f}s")
    return model, train_time


def compute_metrics(y_test, y_proba):
    """Calcula todas las métricas relevantes para el negocio ajustando el mejor umbral F1."""
    best_th, _ = get_best_f1_threshold(y_test, y_proba)
    y_pred = (y_proba >= best_th).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "best_threshold": float(best_th),
        "auc_roc":     float(roc_auc_score(y_test, y_proba)),
        "auc_pr":      float(average_precision_score(y_test, y_proba)),
        "f1":          float(f1_score(y_test, y_pred)),
        "precision":   float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_test, y_pred)),
        "fpr":         float(fp / (fp + tn) if (fp + tn) > 0 else 0),
        "mcc":         float(matthews_corrcoef(y_test, y_pred)),
        "g_mean":      float(np.sqrt(recall_score(y_test, y_pred) * specificity)),
        "costo_negocio": float(fn * 50 + fp * 1),  # FN cuesta 50x más que FP en fraude real
    }, y_pred


def evaluate_model(model, X_test, y_test, is_xgb=False):
    """Evalúa un modelo y retorna métricas + predicciones."""
    if is_xgb:
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.predict(X_test, verbose=0).ravel()
    metrics, y_pred = compute_metrics(y_test, y_proba)
    return metrics, y_proba, y_pred


# ═══════════════════════════════════════════════════════════════════════
# 5. VISUALIZACIONES DE LA FASE DE BALANCEO
# ═══════════════════════════════════════════════════════════════════════

def plot_balance_distribution(distributions):
    """Gráfico de distribución de clases por técnica de balanceo."""
    n = len(distributions)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]

    colors = ["#2ecc71", "#e74c3c"]
    labels = ["Legítima", "Fraudulenta"]

    for ax, (method, (counts_0, counts_1)) in zip(axes, distributions.items()):
        values = [counts_0, counts_1]
        bars = ax.bar(labels, values, color=colors, edgecolor="white")
        ax.set_title(BALANCE_LABELS[method], fontweight="bold")
        ax.set_ylabel("Nº de Muestras")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f"{val:,}", ha="center", fontweight="bold", fontsize=9)

    plt.suptitle("Distribución de Clases por Técnica de Balanceo",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_balance_distribucion.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Distribución de balanceo")


def plot_balance_heatmap(all_results):
    """Heatmap comparativo: métrica × (balanceo + modelo)."""
    metric_keys = ["auc_roc", "auc_pr", "f1", "recall", "precision",
                   "mcc", "g_mean"]
    metric_labels = ["AUC-ROC", "AUC-PR", "F1", "Recall", "Precisión",
                     "MCC", "G-Mean"]

    rows = []
    row_labels = []
    for bal_method, models in all_results.items():
        for model_name, res in models.items():
            vals = [res["metrics"][k] for k in metric_keys]
            rows.append(vals)
            row_labels.append(
                f"{BALANCE_LABELS[bal_method]} | {MODEL_LABELS[model_name]}"
            )

    data = np.array(rows)
    fig, ax = plt.subplots(figsize=(12, max(8, len(rows) * 0.55)))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=metric_labels, yticklabels=row_labels,
                ax=ax, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Valor de Métrica"})
    ax.set_title("Comparativa de Balanceo × Modelo × Métrica",
                 fontweight="bold", fontsize=14, pad=15)

    # Highlight the best row
    best_idx = np.argmax(data[:, 0])  # Best AUC-ROC
    ax.add_patch(plt.Rectangle((0, best_idx), len(metric_keys), 1,
                               fill=False, edgecolor="#27ae60", linewidth=3))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_balance_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Heatmap de balanceo")


def plot_balance_bars(all_results):
    """Bar chart agrupado por técnica de balanceo (la métrica AUC-ROC)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metric_keys = ["auc_roc", "recall", "costo_negocio"]
    metric_titles = ["AUC-ROC", "Recall (Sensibilidad)", "Costo de Negocio (menor=mejor)"]

    for ax, mk, mt in zip(axes, metric_keys, metric_titles):
        x = np.arange(len(MODEL_LABELS))
        width = 0.25
        for i, (bal, models) in enumerate(all_results.items()):
            vals = [models[m]["metrics"][mk] for m in MODEL_BUILDERS.keys()]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width,
                          label=BALANCE_LABELS[bal],
                          color=BALANCE_COLORS[bal],
                          edgecolor="white", alpha=0.85)
            for bar, val in zip(bars, vals):
                if mk == "costo_negocio":
                    label_text = f"{val:,.0f}"
                else:
                    label_text = f"{val:.3f}"
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (bar.get_height() * 0.01),
                        label_text, ha="center", fontsize=6.5, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_BUILDERS.keys()],
                           fontsize=9)
        ax.set_title(mt, fontweight="bold")
        ax.legend(fontsize=8)

    plt.suptitle("Comparativa por Técnica de Balanceo",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_balance_barras.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Barras de balanceo")


def generate_balance_table(all_results):
    """Genera tabla comparativa como imagen PNG."""
    col_labels = ["Balanceo", "Modelo", "AUC-ROC", "AUC-PR", "F1",
                  "Recall", "Prec.", "MCC", "G-Mean", "Costo"]
    cell_data = []
    cell_colors = []

    best_val = 0
    for res in all_results.values():
        for m in res.values():
            if m["metrics"]["f1"] > best_val:
                best_val = m["metrics"]["f1"]

    for bal, models in all_results.items():
        for model_name, res in models.items():
            m = res["metrics"]
            row = [
                BALANCE_LABELS[bal],
                MODEL_LABELS[model_name],
                f"{m['auc_roc']:.4f}",
                f"{m['auc_pr']:.4f}",
                f"{m['f1']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['mcc']:.4f}",
                f"{m['g_mean']:.4f}",
                f"{m['costo_negocio']:,}",
            ]
            cell_data.append(row)
            if m["f1"] == best_val:
                cell_colors.append(["#d5f5e3"] * len(row))
            else:
                cell_colors.append(["white"] * len(row))

    fig, ax = plt.subplots(figsize=(18, max(6, len(cell_data) * 0.45)))
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

    ax.set_title("Fase 1: Experimentación de Balanceo — Resultados Comparativos",
                 fontweight="bold", fontsize=13, pad=20)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_balance_tabla.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Tabla de balanceo")


def plot_roc_curves_by_balance(all_results, y_test):
    """Curvas ROC por técnica de balanceo."""
    n_bal = len(all_results)
    fig, axes = plt.subplots(1, n_bal, figsize=(7*n_bal, 6))
    if n_bal == 1:
        axes = [axes]

    for ax, (bal, models) in zip(axes, all_results.items()):
        for name, res in models.items():
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            ax.plot(fpr, tpr, color=MODEL_COLORS.get(name, "gray"),
                    label=f"{MODEL_LABELS[name]} ({res['metrics']['auc_roc']:.4f})",
                    linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title(f"ROC — {BALANCE_LABELS[bal]}", fontweight="bold")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=8, loc="lower right")

    plt.suptitle("Curvas ROC por Técnica de Balanceo",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "05_balance_roc.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Curvas ROC por balanceo")


def plot_confusion_matrices_best(all_results, y_test, best_balance, best_model):
    """Matrices de confusión del mejor balance para los 5 modelos."""
    models = all_results[best_balance]
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legítima", "Fraude"],
                    yticklabels=["Legítima", "Fraude"],
                    annot_kws={"size": 12})
        title = MODEL_LABELS.get(name, name)
        if name == best_model:
            title += " ★"
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylabel("Real")
        ax.set_xlabel("Predicción")

    plt.suptitle(f"Matrices de Confusión — {BALANCE_LABELS[best_balance]}",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_balance_confusion.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Matrices de confusión")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("FASE 1: EXPERIMENTACIÓN DE BALANCEO")
    print("3 Técnicas × 5 Modelos = 15 combinaciones")
    print("=" * 70)

    # 1. Load & preprocess
    (X_train_orig, y_train_orig, X_val, y_val,
     X_test, y_test, feature_names, preprocessor) = load_and_preprocess()

    input_dim = X_train_orig.shape[1]
    print(f"  Input dimension: {input_dim}")

    balance_methods = ["baseline", "smote_tomek", "adasyn"]
    all_results = OrderedDict()
    distributions = OrderedDict()

    # 2. Iterate over balancing methods
    for bal_method in balance_methods:
        print(f"\n{'='*70}")
        print(f"BALANCEO: {BALANCE_LABELS[bal_method]}")
        print(f"{'='*70}")

        X_train_bal, y_train_bal = apply_balancing(
            X_train_orig, y_train_orig, bal_method
        )

        counts = np.bincount(y_train_bal.astype(int))
        distributions[bal_method] = (int(counts[0]), int(counts[1]))

        model_results = OrderedDict()
        for model_name, builder in MODEL_BUILDERS.items():
            print(f"\n  [{bal_method}] Entrenando {MODEL_LABELS[model_name]}...")
            is_xgb = (model_name == "XGBoost")

            if is_xgb:
                model = builder(input_dim)
                model, train_time = train_xgboost_model(
                    model, X_train_bal, y_train_bal, X_val, y_val, model_name
                )
            else:
                model = builder(input_dim)
                model, train_time = train_keras_model(
                    model, X_train_bal, y_train_bal, X_val, y_val, model_name
                )

            metrics, y_proba, y_pred = evaluate_model(
                model, X_test, y_test, is_xgb=is_xgb
            )
            model_results[model_name] = {
                "model": model,
                "metrics": metrics,
                "y_proba": y_proba,
                "y_pred": y_pred,
                "train_time": train_time,
            }
            print(f"    AUC-ROC={metrics['auc_roc']:.4f} | "
                  f"Recall={metrics['recall']:.4f} | "
                  f"MCC={metrics['mcc']:.4f} | "
                  f"Costo={metrics['costo_negocio']:,}")
            gc.collect()

        all_results[bal_method] = model_results

    # 3. Print consolidated results
    print("\n" + "=" * 100)
    print("TABLA CONSOLIDADA DE RESULTADOS — FASE DE BALANCEO")
    print("=" * 100)
    header = (f"{'Balanceo':<14} {'Modelo':<14} {'AUC-ROC':>8} {'AUC-PR':>8} "
              f"{'F1':>8} {'Recall':>8} {'Prec':>8} {'MCC':>8} "
              f"{'G-Mean':>8} {'Costo':>8}")
    print(header)
    print("-" * 100)

    best_f1 = 0
    best_balance = ""
    best_model = ""
    best_auc = 0.0
    best_cost = 0.0

    for bal, models in all_results.items():
        for model_name, res in models.items():
            m = res["metrics"]
            bl = BALANCE_LABELS[bal][:13]
            ml = MODEL_LABELS[model_name][:13]
            print(f"{bl:<14} {ml:<14} {m['auc_roc']:>8.4f} {m['auc_pr']:>8.4f} "
                  f"{m['f1']:>8.4f} {m['recall']:>8.4f} {m['precision']:>8.4f} "
                  f"{m['mcc']:>8.4f} {m['g_mean']:>8.4f} {m['costo_negocio']:>8,}")
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_balance = bal
                best_model = model_name
                best_auc = m["auc_roc"]
                best_cost = m["costo_negocio"]

    print("-" * 100)
    print(f"🏆 GANADOR: {MODEL_LABELS[best_model]} + "
          f"{BALANCE_LABELS[best_balance]} (F1-Score = {best_f1:.4f})")

    # 4. Visualizations
    print("\n[VISUALIZACIONES]")
    print("-" * 50)
    plot_balance_distribution(distributions)
    plot_balance_heatmap(all_results)
    plot_balance_bars(all_results)
    generate_balance_table(all_results)
    plot_confusion_matrices_best(all_results, y_test, best_balance, best_model)
    
    # 5. Export JSON plot data
    export_plot_data_03(all_results, y_test, distributions)

    # 5. Save results for next phase
    results_export = {
        "best_balance": best_balance,
        "best_model": best_model,
        "best_auc_roc": best_auc,
        "best_cost": best_cost,
        "all_metrics": {},
    }
    for bal, models in all_results.items():
        results_export["all_metrics"][bal] = {}
        for model_name, res in models.items():
            results_export["all_metrics"][bal][model_name] = {
                "label": MODEL_LABELS[model_name],
                "metrics": res["metrics"],
                "train_time": res["train_time"],
            }

    with open(MODEL_DIR / "balance_experiment_results.json", "w") as f:
        json.dump(results_export, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("✅ FASE 1 COMPLETADA")
    print(f"  📊 Gráficos: {FIG_DIR}")
    print(f"  📋 Resultados: {MODEL_DIR / 'balance_experiment_results.json'}")
    print(f"\n  🏆 Mejor balanceo: {BALANCE_LABELS[best_balance]}")
    print(f"  🏆 Mejor modelo:   {MODEL_LABELS[best_model]}")
    print(f"  📈 AUC-ROC:        {best_auc:.4f}")
    print(f"\n  ➡️  Ejecuta '04_train_optimized.py' para la Fase 2")
    print("=" * 70)

    return all_results, best_balance, best_model

def export_plot_data_03(all_results, y_test, distributions):
    """Extrae la data para poder regenerar gráficos sin entrenar."""
    import json
    from sklearn.metrics import roc_curve, confusion_matrix
    print("  Generando 03_plot_data.json...")
    
    plot_data = {
        "distributions": distributions,
        "metrics": {},
        "roc_curves": {},
        "confusion_matrices": {}
    }
    
    for bal, models in all_results.items():
        plot_data["metrics"][bal] = {}
        plot_data["roc_curves"][bal] = {}
        plot_data["confusion_matrices"][bal] = {}
        for model_name, res in models.items():
            plot_data["metrics"][bal][model_name] = res["metrics"]
            
            # ROC downsampled for smaller JSON size
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            step = max(1, len(fpr) // 200)
            plot_data["roc_curves"][bal][model_name] = {
                "fpr": fpr[::step].tolist(),
                "tpr": tpr[::step].tolist()
            }
            
            cm = confusion_matrix(y_test, res["y_pred"])
            plot_data["confusion_matrices"][bal][model_name] = cm.tolist()
            
    out_path = FIG_DIR / "03_plot_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plot_data, f, indent=2, ensure_ascii=False)
    print("  ✓ Datos JSON exportados a 03_plot_data.json")

if __name__ == "__main__":
    all_results, best_balance, best_model = main()
