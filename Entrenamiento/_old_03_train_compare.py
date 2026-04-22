#!/usr/bin/env python3
"""
03_train_compare.py — Pipeline de entrenamiento y comparación de 5 modelos
de Deep Learning para detección de fraude en e-commerce.

Modelos:
  1. DNN (Feed-Forward Dense)
  2. 1D-CNN (Convolutional)
  3. BiLSTM (Bidirectional LSTM)
  4. Autoencoder-Classifier
  5. TabTransformer (Attention-based)

Incluye: Preprocesamiento, balanceo SMOTE, entrenamiento, evaluación,
visualizaciones comparativas, explicabilidad SHAP, y exportación.
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from joblib import dump

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
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

# Estilo gráficos
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.grid": True, "grid.alpha": 0.3,
})

# ═══════════════════════════════════════════════════════════════════════
# 1. CARGA Y PREPROCESAMIENTO
# ═══════════════════════════════════════════════════════════════════════

# -- Features --
NUMERIC_FEATURES = [
    "transaction_amount", "amt_log1p", "hour", "day_of_week", "month",
    "is_weekend", "eci_code", "has_3ds", "city_population", "num_items",
    "has_discount", "num_installments", "previous_failed_attempts",
    "is_new_customer", "days_since_first_purchase", "avg_historical_amount",
    "is_high_risk_hour", "amount_deviation",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
]
CATEGORICAL_FEATURES = [
    "card_brand", "card_type", "issuer_bank", "payment_channel",
    "customer_region", "category",
]
TARGET = "is_fraud"


def load_and_preprocess():
    """Carga datos, preprocesa, y divide en train/val/test."""
    print("\n[1] CARGA Y PREPROCESAMIENTO")
    print("-" * 50)

    df = pd.read_csv(DATA_DIR / "fraud_ecommerce_dataset.csv")
    print(f"  Dataset: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"  Fraude: {df[TARGET].sum():,} ({df[TARGET].mean()*100:.1f}%)")

    # Separar features y target
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].values

    # Preprocesador: StandardScaler para numéricos + OneHot para categóricos
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SKPipeline([("scaler", StandardScaler())]), NUMERIC_FEATURES),
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
        X_temp, y_temp, test_size=0.176, random_state=SEED, stratify=y_temp  # 0.176 of 85% ≈ 15%
    )

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Fit preprocessor on train only
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc   = preprocessor.transform(X_val)
    X_test_proc  = preprocessor.transform(X_test)

    # Feature names after preprocessing
    num_names = NUMERIC_FEATURES
    cat_names = list(preprocessor.named_transformers_["cat"]
                     .named_steps["onehot"]
                     .get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = num_names + cat_names

    print(f"  Features después de preprocessing: {len(feature_names)}")

    # SMOTE en train solamente
    print(f"  Aplicando SMOTE...")
    print(f"    Antes: {np.bincount(y_train)}")
    smote = SMOTE(random_state=SEED, sampling_strategy=0.4)  # 40% ratio
    X_train_proc, y_train = smote.fit_resample(X_train_proc, y_train)
    print(f"    Después: {np.bincount(y_train)}")

    # Guardar preprocesador y names
    dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    return (X_train_proc, y_train, X_val_proc, y_val,
            X_test_proc, y_test, feature_names, preprocessor)


# ═══════════════════════════════════════════════════════════════════════
# 2. DEFINICIÓN DE MODELOS
# ═══════════════════════════════════════════════════════════════════════

def build_dnn(input_dim):
    """Modelo 1: Deep Neural Network (Feed-Forward)"""
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ], name="DNN")
    return model


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


def build_bilstm(input_dim):
    """Modelo 3: Bidirectional LSTM"""
    inp = layers.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inp)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out, name="BiLSTM")


def build_autoencoder_classifier(input_dim):
    """Modelo 4: Autoencoder + Classification Head"""
    inp = layers.Input(shape=(input_dim,))
    # Encoder
    enc = layers.Dense(128, activation="relu")(inp)
    enc = layers.BatchNormalization()(enc)
    enc = layers.Dropout(0.3)(enc)
    enc = layers.Dense(64, activation="relu")(enc)
    enc = layers.BatchNormalization()(enc)
    bottleneck = layers.Dense(24, activation="relu", name="bottleneck")(enc)
    # Decoder (reconstruction)
    dec = layers.Dense(64, activation="relu")(bottleneck)
    dec = layers.Dense(128, activation="relu")(dec)
    reconstruction = layers.Dense(input_dim, activation="linear", name="reconstruction")(dec)
    # Classifier head
    cls = layers.Dense(32, activation="relu")(bottleneck)
    cls = layers.Dropout(0.2)(cls)
    classification = layers.Dense(1, activation="sigmoid", name="classification")(cls)

    model = Model(inp, classification, name="AutoEncoder_Clf")
    return model


def build_tab_transformer(input_dim):
    """Modelo 5: Tab Transformer (Attention-based for tabular data)"""
    inp = layers.Input(shape=(input_dim,))

    # Project to embedding dimension
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Reshape((1, 64))(x)  # (batch, 1, 64)

    # Duplicate for multi-head attention (simulate multiple tokens)
    # Split features into groups as tokens
    tokens = layers.Dense(64 * 4)(inp)  # 4 tokens
    tokens = layers.Reshape((4, 64))(tokens)

    # Multi-head self-attention blocks (2 layers)
    for _ in range(2):
        # Self-attention
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(tokens, tokens)
        attn = layers.Dropout(0.15)(attn)
        tokens = layers.LayerNormalization()(tokens + attn)
        # FFN
        ffn = layers.Dense(128, activation="gelu")(tokens)
        ffn = layers.Dense(64)(ffn)
        ffn = layers.Dropout(0.15)(ffn)
        tokens = layers.LayerNormalization()(tokens + ffn)

    # Pool and classify
    x = layers.GlobalAveragePooling1D()(tokens)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    return Model(inp, out, name="TabTransformer")


# ═══════════════════════════════════════════════════════════════════════
# 3. ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════

def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=100, batch_size=256):
    """Entrena un modelo con early stopping y LR scheduling."""
    print(f"\n  Entrenando {model_name}...")

    # Class weights
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    weight_ratio = n_neg / n_pos
    class_weight = {0: 1.0, 1: min(weight_ratio, 5.0)}

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_auc", patience=12, restore_best_weights=True, mode="max"
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )
    train_time = time.time() - start

    # Inference time (avg per sample)
    start_inf = time.time()
    _ = model.predict(X_val[:1000], verbose=0)
    inf_time = (time.time() - start_inf) / 1000 * 1000  # ms per sample

    best_epoch = np.argmax(history.history.get("val_auc", [0]))
    print(f"    ✓ {model_name} entrenado en {train_time:.1f}s "
          f"({best_epoch+1} epochs) — Inferencia: {inf_time:.3f}ms/sample")

    return model, history, train_time, inf_time


def evaluate_model(model, X_test, y_test):
    """Evalúa un modelo y retorna métricas."""
    y_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_test, y_proba),
        "auc_pr": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "fpr": (confusion_matrix(y_test, y_pred)[0, 1] /
                confusion_matrix(y_test, y_pred)[0].sum()),
    }
    return metrics, y_proba, y_pred


# ═══════════════════════════════════════════════════════════════════════
# 4. VISUALIZACIONES COMPARATIVAS
# ═══════════════════════════════════════════════════════════════════════

MODEL_COLORS = {
    "DNN": "#3498db",
    "CNN_1D": "#e74c3c",
    "BiLSTM": "#2ecc71",
    "AutoEncoder_Clf": "#9b59b6",
    "TabTransformer": "#f39c12",
}

MODEL_LABELS = {
    "DNN": "DNN",
    "CNN_1D": "CNN-1D",
    "BiLSTM": "BiLSTM",
    "AutoEncoder_Clf": "Autoencoder",
    "TabTransformer": "TabTransformer",
}


def plot_roc_curves(results, y_test):
    """Curvas ROC superpuestas de los 5 modelos."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr,
                color=MODEL_COLORS.get(name, "gray"),
                label=f"{MODEL_LABELS.get(name, name)} (AUC={res['metrics']['auc_roc']:.4f})",
                linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.set_title("Curvas ROC — Comparación de 5 Modelos Deep Learning", fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_roc_curves.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Curvas ROC")


def plot_pr_curves(results, y_test):
    """Curvas Precision-Recall superpuestas."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        ax.plot(rec, prec,
                color=MODEL_COLORS.get(name, "gray"),
                label=f"{MODEL_LABELS.get(name, name)} (AP={res['metrics']['auc_pr']:.4f})",
                linewidth=2)
    baseline = y_test.mean()
    ax.axhline(baseline, color="gray", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall (Sensibilidad)")
    ax.set_ylabel("Precisión")
    ax.set_title("Curvas Precision-Recall — Comparación de Modelos", fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_pr_curves.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Curvas Precision-Recall")


def plot_metrics_comparison(results):
    """Bar chart comparativo de métricas."""
    metric_names = ["auc_roc", "auc_pr", "f1", "precision", "recall"]
    metric_labels = ["AUC-ROC", "AUC-PR", "F1-Score", "Precisión", "Recall"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metric_names))
    width = 0.15
    offsets = np.linspace(-2*width, 2*width, len(results))

    for i, (name, res) in enumerate(results.items()):
        vals = [res["metrics"][m] for m in metric_names]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=MODEL_LABELS.get(name, name),
                      color=MODEL_COLORS.get(name, "gray"),
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Valor")
    ax.set_title("Comparación de Métricas — 5 Modelos Deep Learning", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.axhline(0.97, color="gray", linestyle=":", alpha=0.5, label="Meta AUC=0.97")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_metricas_comparacion.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Comparación de métricas")


def plot_confusion_matrices(results, y_test):
    """Confusion matrices lado a lado."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legítima", "Fraude"],
                    yticklabels=["Legítima", "Fraude"],
                    annot_kws={"size": 12})
        ax.set_title(MODEL_LABELS.get(name, name), fontweight="bold", fontsize=11)
        ax.set_ylabel("Real")
        ax.set_xlabel("Predicción")

    plt.suptitle("Matrices de Confusión", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_confusion_matrices.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Matrices de confusión")


def plot_training_curves(results):
    """Training loss and AUC curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax = axes[0]
    for name, res in results.items():
        h = res["history"].history
        ax.plot(h["loss"], color=MODEL_COLORS.get(name, "gray"),
                label=f"{MODEL_LABELS.get(name, name)} (train)", linewidth=1.5)
        ax.plot(h["val_loss"], color=MODEL_COLORS.get(name, "gray"),
                linestyle="--", alpha=0.6, linewidth=1)
    ax.set_title("Curvas de Pérdida (Entrenamiento)", fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("Binary Crossentropy")
    ax.legend(fontsize=8, ncol=2)

    # AUC curves
    ax = axes[1]
    for name, res in results.items():
        h = res["history"].history
        if "auc" in h:
            ax.plot(h["auc"], color=MODEL_COLORS.get(name, "gray"),
                    label=f"{MODEL_LABELS.get(name, name)} (train)", linewidth=1.5)
        if "val_auc" in h:
            ax.plot(h["val_auc"], color=MODEL_COLORS.get(name, "gray"),
                    linestyle="--", alpha=0.6, linewidth=1)
    ax.set_title("Curvas AUC (Entrenamiento)", fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("AUC-ROC")
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "05_training_curves.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Curvas de entrenamiento")


def plot_inference_time(results):
    """Tiempo de entrenamiento e inferencia."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    names = [MODEL_LABELS.get(n, n) for n in results.keys()]
    colors = [MODEL_COLORS.get(n, "gray") for n in results.keys()]

    # Training time
    train_times = [res["train_time"] for res in results.values()]
    ax1.barh(names, train_times, color=colors, edgecolor="white")
    ax1.set_title("Tiempo de Entrenamiento (s)", fontweight="bold")
    ax1.set_xlabel("Segundos")
    for i, val in enumerate(train_times):
        ax1.text(val + 0.5, i, f"{val:.1f}s", va="center", fontsize=10)

    # Inference time
    inf_times = [res["inf_time"] for res in results.values()]
    ax2.barh(names, inf_times, color=colors, edgecolor="white")
    ax2.set_title("Tiempo de Inferencia (ms/muestra)", fontweight="bold")
    ax2.set_xlabel("Milisegundos")
    for i, val in enumerate(inf_times):
        ax2.text(val + 0.001, i, f"{val:.3f}ms", va="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_tiempos.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Tiempos de entrenamiento/inferencia")


def plot_smote_effect(y_original, y_smote):
    """Distribución antes/después de SMOTE."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#2ecc71", "#e74c3c"]
    labels = ["Legítima", "Fraudulenta"]

    # Antes
    before = np.bincount(y_original)
    ax1.bar(labels, before, color=colors, edgecolor="white")
    ax1.set_title("Antes de SMOTE", fontweight="bold")
    ax1.set_ylabel("Número de Muestras")
    for i, val in enumerate(before):
        ax1.text(i, val + 50, f"{val:,}", ha="center", fontweight="bold")

    # Después
    after = np.bincount(y_smote)
    ax2.bar(labels, after, color=colors, edgecolor="white")
    ax2.set_title("Después de SMOTE (ratio=0.4)", fontweight="bold")
    ax2.set_ylabel("Número de Muestras")
    for i, val in enumerate(after):
        ax2.text(i, val + 50, f"{val:,}", ha="center", fontweight="bold")

    plt.suptitle("Efecto del Balanceo de Clases con SMOTE", fontweight="bold",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_smote_balanceo.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Efecto de SMOTE")


# ═══════════════════════════════════════════════════════════════════════
# 5. EXPLICABILIDAD (SHAP)
# ═══════════════════════════════════════════════════════════════════════

def run_shap_analysis(best_model, best_name, X_test, y_test, feature_names):
    """SHAP analysis para el mejor modelo."""
    print(f"\n[6] ANÁLISIS SHAP — {best_name}")
    print("-" * 50)
    import shap

    # Background samples
    bg_size = min(200, len(X_test))
    bg_indices = np.random.choice(len(X_test), bg_size, replace=False)
    background = X_test[bg_indices]

    # Save background for deployment
    np.save(MODEL_DIR / "background.npy", background)

    # Use KernelExplainer (works with any model)
    print("  Calculando SHAP values (esto puede tomar unos minutos)...")
    try:
        explainer = shap.DeepExplainer(best_model, background)
        test_sample = X_test[:500]
        shap_values = explainer.shap_values(test_sample)
    except Exception:
        print("  DeepExplainer falló, usando KernelExplainer...")
        predict_fn = lambda x: best_model.predict(x, verbose=0).ravel()
        explainer = shap.KernelExplainer(predict_fn, background)
        test_sample = X_test[:200]
        shap_values = explainer.shap_values(test_sample, nsamples=100)

    # Process shap values
    if isinstance(shap_values, list):
        sv = shap_values[0]
    else:
        sv = shap_values
    sv = np.squeeze(sv)

    # Truncate feature names for display
    short_names = []
    for fn in feature_names:
        fn = fn.replace("num__", "").replace("cat__", "")
        if len(fn) > 25:
            fn = fn[:22] + "..."
        short_names.append(fn)

    # 1. Summary bar plot
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
    ax.set_title(f"Importancia de Features — {MODEL_LABELS.get(best_name, best_name)} (SHAP)",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(XAI_DIR / "01_shap_importancia.png", bbox_inches="tight")
    plt.close()
    print("  ✓ SHAP Feature Importance")

    # 2. SHAP summary beeswarm (matplotlib fallback)
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv[:, sorted_idx], test_sample[:len(sv), sorted_idx],
                         feature_names=[short_names[i] for i in sorted_idx],
                         show=False, max_display=20)
        plt.title(f"SHAP Summary Plot — {MODEL_LABELS.get(best_name, best_name)}",
                  fontweight="bold")
        plt.tight_layout()
        plt.savefig(XAI_DIR / "02_shap_summary.png", bbox_inches="tight", dpi=300)
        plt.close("all")
        print("  ✓ SHAP Summary Plot")
    except Exception as e:
        print(f"  ⚠ SHAP Summary Plot falló: {e}")

    # Build group map for deployment
    group_map = {}
    for i, fn in enumerate(feature_names):
        base = fn.split("_")[0] if "_" in fn else fn
        if base not in group_map:
            group_map[base] = []
        group_map[base].append(fn)

    with open(MODEL_DIR / "group_map.json", "w") as f:
        json.dump(group_map, f, indent=2)

    return sv


# ═══════════════════════════════════════════════════════════════════════
# 6. TABLA RESUMEN
# ═══════════════════════════════════════════════════════════════════════

def print_results_table(results):
    """Imprime tabla comparativa de resultados."""
    print("\n" + "=" * 90)
    print("TABLA COMPARATIVA DE RESULTADOS")
    print("=" * 90)
    header = f"{'Modelo':<18} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'FPR':>8} {'Train(s)':>9} {'Inf(ms)':>8}"
    print(header)
    print("-" * 90)

    best_auc = 0
    best_name = ""
    for name, res in results.items():
        m = res["metrics"]
        label = MODEL_LABELS.get(name, name)
        print(f"{label:<18} {m['auc_roc']:>8.4f} {m['auc_pr']:>8.4f} {m['f1']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['fpr']:>8.4f} "
              f"{res['train_time']:>8.1f}s {res['inf_time']:>7.3f}")
        if m["auc_roc"] > best_auc:
            best_auc = m["auc_roc"]
            best_name = name

    print("-" * 90)
    print(f"🏆 Mejor modelo: {MODEL_LABELS.get(best_name, best_name)} (AUC-ROC = {best_auc:.4f})")
    return best_name


def generate_results_table_figure(results):
    """Genera tabla de resultados como imagen."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    col_labels = ["Modelo", "AUC-ROC", "AUC-PR", "F1-Score", "Precisión", "Recall", "FPR", "Tiempo (s)"]
    cell_data = []
    cell_colors = []

    best_auc = max(r["metrics"]["auc_roc"] for r in results.values())

    for name, res in results.items():
        m = res["metrics"]
        row = [
            MODEL_LABELS.get(name, name),
            f"{m['auc_roc']:.4f}",
            f"{m['auc_pr']:.4f}",
            f"{m['f1']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['fpr']:.4f}",
            f"{res['train_time']:.1f}",
        ]
        cell_data.append(row)
        if m["auc_roc"] == best_auc:
            cell_colors.append(["#d5f5e3"] * len(row))
        else:
            cell_colors.append(["white"] * len(row))

    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     cellColours=cell_colors,
                     cellLoc="center", loc="center",
                     colColours=["#3498db"] * len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Header color
    for j in range(len(col_labels)):
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Tabla Comparativa de Modelos Deep Learning para Detección de Fraude",
                 fontweight="bold", fontsize=13, pad=20)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "08_tabla_resultados.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Tabla de resultados")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PIPELINE DE ENTRENAMIENTO — Detección de Fraude E-commerce")
    print("5 Modelos Deep Learning")
    print("=" * 70)

    # 1. Load & preprocess
    (X_train, y_train, X_val, y_val,
     X_test, y_test, feature_names, preprocessor) = load_and_preprocess()

    input_dim = X_train.shape[1]
    print(f"  Input dimension: {input_dim}")

    # Save original y_train for SMOTE visualization (before SMOTE was applied in load_and_preprocess)
    # Re-compute for visualization
    df_temp = pd.read_csv(DATA_DIR / "fraud_ecommerce_dataset.csv")
    y_orig_train = df_temp[TARGET].values
    X_temp_all, _, y_temp_all, _ = train_test_split(
        df_temp[NUMERIC_FEATURES + CATEGORICAL_FEATURES], y_orig_train,
        test_size=0.15, random_state=SEED, stratify=y_orig_train
    )
    X_temp_tr, _, y_temp_tr, _ = train_test_split(
        X_temp_all, y_temp_all, test_size=0.176, random_state=SEED, stratify=y_temp_all
    )
    plot_smote_effect(y_temp_tr, y_train)

    # 2. Build & train models
    print("\n[2] CONSTRUCCIÓN Y ENTRENAMIENTO DE MODELOS")
    print("-" * 50)

    builders = OrderedDict([
        ("DNN",              build_dnn),
        ("CNN_1D",           build_cnn),
        ("BiLSTM",           build_bilstm),
        ("AutoEncoder_Clf",  build_autoencoder_classifier),
        ("TabTransformer",   build_tab_transformer),
    ])

    results = OrderedDict()
    for name, builder in builders.items():
        model = builder(input_dim)
        model, history, train_time, inf_time = train_model(
            model, X_train, y_train, X_val, y_val, name
        )
        metrics, y_proba, y_pred = evaluate_model(model, X_test, y_test)
        results[name] = {
            "model": model,
            "history": history,
            "metrics": metrics,
            "y_proba": y_proba,
            "y_pred": y_pred,
            "train_time": train_time,
            "inf_time": inf_time,
        }
        print(f"    AUC-ROC={metrics['auc_roc']:.4f} | AUC-PR={metrics['auc_pr']:.4f} | "
              f"F1={metrics['f1']:.4f} | Recall={metrics['recall']:.4f}")
        # Clear memory
        gc.collect()

    # 3. Results
    print("\n[3] RESULTADOS Y COMPARACIÓN")
    print("-" * 50)
    best_name = print_results_table(results)

    # 4. Visualizations
    print("\n[4] GENERANDO VISUALIZACIONES")
    print("-" * 50)
    plot_roc_curves(results, y_test)
    plot_pr_curves(results, y_test)
    plot_metrics_comparison(results)
    plot_confusion_matrices(results, y_test)
    plot_training_curves(results)
    plot_inference_time(results)
    generate_results_table_figure(results)

    # 5. Save best model
    print("\n[5] GUARDANDO MEJOR MODELO")
    print("-" * 50)
    best_model = results[best_name]["model"]
    model_path = MODEL_DIR / "best_model.keras"
    best_model.save(model_path)
    print(f"  ✓ Modelo guardado: {model_path}")
    print(f"  ✓ Modelo: {MODEL_LABELS.get(best_name, best_name)}")
    print(f"  ✓ AUC-ROC: {results[best_name]['metrics']['auc_roc']:.4f}")

    # Save all models
    for name, res in results.items():
        res["model"].save(MODEL_DIR / f"{name}.keras")

    # Save metrics
    metrics_export = {}
    for name, res in results.items():
        metrics_export[name] = {
            "label": MODEL_LABELS.get(name, name),
            "metrics": res["metrics"],
            "train_time": res["train_time"],
            "inf_time": res["inf_time"],
        }
    with open(MODEL_DIR / "comparison_metrics.json", "w") as f:
        json.dump(metrics_export, f, indent=2, default=str)

    # 6. SHAP
    try:
        shap_values = run_shap_analysis(best_model, best_name, X_test, y_test, feature_names)
    except Exception as e:
        print(f"  ⚠ SHAP análisis falló: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETADO")
    print(f"  📊 Gráficos comparativos: {FIG_DIR}")
    print(f"  🧠 Modelos guardados: {MODEL_DIR}")
    print(f"  🔍 Explicabilidad: {XAI_DIR}")
    print(f"  📈 EDA: {BASE_DIR / 'outputs' / 'eda'}")
    print(f"\n  🏆 Mejor modelo: {MODEL_LABELS.get(best_name, best_name)}")
    bm = results[best_name]["metrics"]
    print(f"     AUC-ROC:    {bm['auc_roc']:.4f}")
    print(f"     AUC-PR:     {bm['auc_pr']:.4f}")
    print(f"     F1-Score:   {bm['f1']:.4f}")
    print(f"     Precisión:  {bm['precision']:.4f}")
    print(f"     Recall:     {bm['recall']:.4f}")
    print("=" * 70)

    return results, best_name


if __name__ == "__main__":
    results, best_name = main()
