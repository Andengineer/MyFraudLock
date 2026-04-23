#!/usr/bin/env python3
"""
02_eda_analysis.py — Análisis Exploratorio de Datos para dataset de fraude.
Genera gráficos de alta calidad en español para paper/tesis.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ─── Configuración ─────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "outputs" / "data"
EDA_DIR  = BASE_DIR / "outputs" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# Estilo visual profesional
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Paleta personalizada
PAL_FRAUD = {"Legítima": "#2ecc71", "Fraudulenta": "#e74c3c"}
PAL_SEQ = sns.color_palette("coolwarm", 12)
COL_PRIMARY = "#3498db"
COL_FRAUD = "#e74c3c"
COL_LEGIT = "#2ecc71"

def load_data():
    path = DATA_DIR / "fraud_ecommerce_dataset.csv"
    df = pd.read_csv(path)
    df["label"] = df["is_fraud"].map({0: "Legítima", 1: "Fraudulenta"})
    print(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df

# ═══════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════

def plot_class_distribution(df):
    """1. Distribución de clases (balance del dataset)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    counts = df["is_fraud"].value_counts().sort_index()
    labels = ["Legítima", "Fraudulenta"]
    colors = [COL_LEGIT, COL_FRAUD]
    
    # Bar chart
    bars = ax1.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", fontweight="bold", fontsize=12)
    ax1.set_title("Distribución de Clases", fontweight="bold")
    ax1.set_ylabel("Número de Transacciones")
    ax1.set_ylim(0, counts.max() * 1.15)
    
    # Pie chart
    ax2.pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.85,
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2))
    ax2.set_title("Proporción de Fraude", fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(EDA_DIR / "01_distribucion_clases.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Distribución de clases")


def plot_amount_distribution(df):
    """2. Distribución de montos por clase"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Histograma superpuesto
    ax = axes[0]
    for label, color in PAL_FRAUD.items():
        subset = df[df["label"] == label]["transaction_amount"]
        ax.hist(subset, bins=60, alpha=0.6, color=color, label=label, density=True)
    ax.set_title("Distribución de Montos (PEN)", fontweight="bold")
    ax.set_xlabel("Monto (S/)")
    ax.set_ylabel("Densidad")
    ax.legend()
    ax.set_xlim(0, 2000)
    
    # Box plot
    ax = axes[1]
    data_box = [df[df["is_fraud"]==0]["transaction_amount"],
                df[df["is_fraud"]==1]["transaction_amount"]]
    bp = ax.boxplot(data_box, labels=["Legítima", "Fraudulenta"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor(COL_LEGIT)
    bp["boxes"][1].set_facecolor(COL_FRAUD)
    ax.set_title("Comparación de Montos", fontweight="bold")
    ax.set_ylabel("Monto (S/)")
    
    # Violin plot
    ax = axes[2]
    sns.violinplot(data=df, x="label", y="amt_log1p", palette=PAL_FRAUD,
                   ax=ax, inner="quartile", cut=0)
    ax.set_title("Log(Monto+1) por Clase", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("log(1 + monto)")
    
    plt.tight_layout()
    fig.savefig(EDA_DIR / "02_distribucion_montos.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Distribución de montos")


def plot_temporal_patterns(df):
    """3. Patrones temporales"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Transacciones por hora
    ax = axes[0, 0]
    for label, color in PAL_FRAUD.items():
        sub = df[df["label"] == label]
        counts_h = sub.groupby("hour").size()
        counts_h = counts_h.reindex(range(24), fill_value=0)
        if label == "Fraudulenta":
            counts_h = counts_h * (len(df[df["is_fraud"]==0]) / len(sub))  # Escalar
        ax.plot(counts_h.index, counts_h.values, color=color, label=label,
                linewidth=2, marker="o", markersize=4)
    ax.set_title("Distribución Horaria (escalada)", fontweight="bold")
    ax.set_xlabel("Hora del Día")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    ax.set_xticks(range(0, 24, 2))
    
    # Heatmap hora vs día de semana
    ax = axes[0, 1]
    day_labels = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    pivot = df[df["is_fraud"]==1].groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
    pivot.index = [day_labels[i] for i in pivot.index]
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "# Fraudes"})
    ax.set_title("Mapa de Calor: Fraudes por Hora y Día", fontweight="bold")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Día de Semana")
    
    # Fraude por día de semana
    ax = axes[1, 0]
    fraud_day = df.groupby("day_of_week")["is_fraud"].mean() * 100
    bars = ax.bar(day_labels, fraud_day.values, color=sns.color_palette("RdYlGn_r", 7),
                  edgecolor="white", linewidth=1)
    ax.set_title("Tasa de Fraude por Día de Semana (%)", fontweight="bold")
    ax.set_ylabel("Tasa de Fraude (%)")
    for bar, val in zip(bars, fraud_day.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.1f}%", ha="center", fontsize=9)
    
    # Fraude por mes
    ax = axes[1, 1]
    month_labels = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    fraud_month = df.groupby("month")["is_fraud"].mean() * 100
    ax.bar(month_labels, fraud_month.values,
           color=sns.color_palette("coolwarm", 12), edgecolor="white", linewidth=1)
    ax.set_title("Tasa de Fraude por Mes (%)", fontweight="bold")
    ax.set_ylabel("Tasa de Fraude (%)")
    
    plt.tight_layout()
    fig.savefig(EDA_DIR / "03_patrones_temporales.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Patrones temporales")


def plot_categorical_analysis(df):
    """4. Análisis de variables categóricas"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    cats = [
        ("card_brand", "Marca de Tarjeta", axes[0, 0]),
        ("card_type", "Tipo de Tarjeta", axes[0, 1]),
        ("payment_channel", "Canal de Pago", axes[0, 2]),
        ("issuer_bank", "Banco Emisor (Top 8)", axes[1, 0]),
        ("customer_region", "Región (Top 8)", axes[1, 1]),
        ("category", "Categoría de Producto", axes[1, 2]),
    ]
    
    for col, title, ax in cats:
        # Top N categorías
        top_n = 8
        top_cats = df[col].value_counts().head(top_n).index
        sub = df[df[col].isin(top_cats)]
        
        fraud_rate = sub.groupby(col)["is_fraud"].mean().sort_values(ascending=True)
        fraud_rate = fraud_rate[fraud_rate.index.isin(top_cats)]
        
        colors = [COL_FRAUD if v > df["is_fraud"].mean() else COL_LEGIT
                  for v in fraud_rate.values]
        
        fraud_rate_pct = fraud_rate * 100
        bars = ax.barh(range(len(fraud_rate_pct)), fraud_rate_pct.values,
                       color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(fraud_rate_pct)))
        ax.set_yticklabels(fraud_rate_pct.index, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Tasa de Fraude (%)")
        ax.axvline(df["is_fraud"].mean() * 100, color="gray", linestyle="--",
                   alpha=0.7, label="Promedio")
        
        for bar, val in zip(bars, fraud_rate_pct.values):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=8)
    
    plt.tight_layout()
    fig.savefig(EDA_DIR / "04_analisis_categorico.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Análisis categórico")


def plot_fraud_types(df):
    """5. Distribución de tipos de fraude"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    fraud_df = df[df["is_fraud"] == 1]
    type_counts = fraud_df["fraud_type"].value_counts()
    type_labels_map = {
        "stolen_card": "Tarjeta Robada",
        "friendly_fraud": "Fraude Amistoso",
        "card_testing": "Prueba de Tarjeta",
        "account_takeover": "Robo de Cuenta",
        "bot_automated": "Bot Automatizado",
        "triangulation": "Triangulación",
    }
    labels = [type_labels_map.get(k, k) for k in type_counts.index]
    colors = sns.color_palette("Set2", len(type_counts))
    
    # Pie chart
    ax1.pie(type_counts.values, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=140, pctdistance=0.85,
            textprops={"fontsize": 10})
    ax1.set_title("Distribución de Tipos de Fraude", fontweight="bold")
    
    # Monto promedio por tipo
    avg_amt = fraud_df.groupby("fraud_type")["transaction_amount"].mean().sort_values()
    labels2 = [type_labels_map.get(k, k) for k in avg_amt.index]
    ax2.barh(labels2, avg_amt.values, color=sns.color_palette("Reds_r", len(avg_amt)),
             edgecolor="white")
    ax2.set_title("Monto Promedio por Tipo de Fraude", fontweight="bold")
    ax2.set_xlabel("Monto Promedio (S/)")
    for i, val in enumerate(avg_amt.values):
        ax2.text(val + 10, i, f"S/ {val:,.0f}", va="center", fontsize=9)
    
    plt.tight_layout()
    fig.savefig(EDA_DIR / "05_tipos_fraude.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Tipos de fraude")


def plot_correlation_matrix(df):
    """6. Matriz de correlación (features numéricas)"""
    num_cols = ["transaction_amount", "amt_log1p", "hour", "day_of_week", "month",
                "is_weekend", "eci_code", "has_3ds", "city_population", "num_items",
                "has_discount", "num_installments", "previous_failed_attempts",
                "is_new_customer", "days_since_first_purchase", "avg_historical_amount",
                "is_high_risk_hour", "amount_deviation", "session_duration_minutes", 
                "interaction_velocity", "device_telemetry_1", "device_telemetry_2", 
                "device_telemetry_3", "device_telemetry_4", "device_telemetry_5", "is_fraud"]
    
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()
    
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=[c.replace("_", "\n") for c in num_cols],
                yticklabels=[c.replace("_", " ") for c in num_cols],
                annot_kws={"size": 7})
    ax.set_title("Matriz de Correlación — Features Numéricas", fontweight="bold", fontsize=14)
    
    plt.tight_layout()
    fig.savefig(EDA_DIR / "06_correlacion.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Matriz de correlación")


def plot_eci_analysis(df):
    """7. Análisis de autenticación ECI (3D Secure)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ECI distribution by fraud
    eci_data = df.groupby(["eci_code", "label"]).size().unstack(fill_value=0)
    eci_data_pct = eci_data.div(eci_data.sum(axis=1), axis=0) * 100
    
    eci_data_pct.plot(kind="bar", stacked=True, ax=ax1,
                      color=[COL_LEGIT, COL_FRAUD], edgecolor="white")
    ax1.set_title("Composición por Código ECI", fontweight="bold")
    ax1.set_xlabel("Código ECI")
    ax1.set_ylabel("Porcentaje (%)")
    ax1.legend(title="Tipo")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    
    # 3DS vs no 3DS fraud rate
    threeds_fraud = df.groupby("has_3ds")["is_fraud"].mean() * 100
    bars = ax2.bar(["Sin 3DS", "Con 3DS"], threeds_fraud.values,
                   color=[COL_FRAUD, COL_LEGIT], edgecolor="white", linewidth=1.5)
    ax2.set_title("Tasa de Fraude según Autenticación 3DS", fontweight="bold")
    ax2.set_ylabel("Tasa de Fraude (%)")
    for bar, val in zip(bars, threeds_fraud.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}%", ha="center", fontweight="bold", fontsize=12)
    
    plt.tight_layout()
    fig.savefig(EDA_DIR / "07_analisis_eci.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Análisis ECI / 3D Secure")


def plot_behavioral_features(df):
    """8. Features comportamentales"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Intentos previos fallidos
    ax = axes[0, 0]
    for label, color in PAL_FRAUD.items():
        sub = df[df["label"] == label]
        counts = sub["previous_failed_attempts"].value_counts().sort_index()
        pcts = counts / len(sub) * 100
        ax.bar(pcts.index + (-0.15 if label == "Legítima" else 0.15),
               pcts.values, width=0.3, color=color, label=label,
               edgecolor="white")
    ax.set_title("Intentos Fallidos Previos", fontweight="bold")
    ax.set_xlabel("Número de Intentos")
    ax.set_ylabel("Porcentaje (%)")
    ax.legend()
    
    # Nuevo cliente
    ax = axes[0, 1]
    new_fraud = df.groupby("is_new_customer")["is_fraud"].mean() * 100
    bars = ax.bar(["Cliente Recurrente", "Nuevo Cliente"], new_fraud.values,
                  color=[COL_LEGIT, COL_FRAUD], edgecolor="white")
    ax.set_title("Tasa de Fraude por Tipo de Cliente", fontweight="bold")
    ax.set_ylabel("Tasa de Fraude (%)")
    for bar, val in zip(bars, new_fraud.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}%", ha="center", fontweight="bold")
    
    # Desviación de monto respecto al historial
    ax = axes[1, 0]
    for label, color in PAL_FRAUD.items():
        sub = df[df["label"] == label]["amount_deviation"].clip(-5, 20)
        ax.hist(sub, bins=50, alpha=0.6, color=color, label=label, density=True)
    ax.set_title("Desviación del Monto vs Historial", fontweight="bold")
    ax.set_xlabel("Desviación (ratio)")
    ax.set_ylabel("Densidad")
    ax.legend()
    
    # Número de cuotas
    ax = axes[1, 1]
    inst_fraud = df.groupby("num_installments")["is_fraud"].mean() * 100
    ax.bar(inst_fraud.index.astype(str), inst_fraud.values,
           color=sns.color_palette("YlOrRd", len(inst_fraud)), edgecolor="white")
    ax.set_title("Tasa de Fraude por Nro. de Cuotas", fontweight="bold")
    ax.set_xlabel("Número de Cuotas")
    ax.set_ylabel("Tasa de Fraude (%)")
    
    plt.tight_layout()
    fig.savefig(EDA_DIR / "08_features_comportamentales.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Features comportamentales")


def plot_geographic_analysis(df):
    """9. Análisis geográfico"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top 10 regiones por volumen
    top_regions = df["customer_region"].value_counts().head(10)
    reg_labels = {
        "lima": "Lima", "arequipa": "Arequipa", "piura": "Piura",
        "cusco": "Cusco", "junin": "Junín", "lambayeque": "Lambayeque",
        "la_libertad": "La Libertad", "callao": "Callao", "ica": "Ica",
        "san_martin": "San Martín", "tacna": "Tacna",
        "ancash": "Áncash", "cajamarca": "Cajamarca",
        "loreto": "Loreto", "ucayali": "Ucayali", "huanuco": "Huánuco",
    }
    labels = [reg_labels.get(r, r) for r in top_regions.index]
    ax1.barh(labels[::-1], top_regions.values[::-1],
             color=sns.color_palette("Blues_r", 10), edgecolor="white")
    ax1.set_title("Top 10 Regiones por Volumen", fontweight="bold")
    ax1.set_xlabel("Número de Transacciones")
    
    # Tasa de fraude por región
    region_fraud = df.groupby("customer_region")["is_fraud"].agg(["mean", "count"])
    region_fraud = region_fraud[region_fraud["count"] > 100]  # Mín 100 transacciones
    region_fraud = region_fraud.sort_values("mean", ascending=True)
    labels2 = [reg_labels.get(r, r) for r in region_fraud.index]
    
    colors = [COL_FRAUD if v > df["is_fraud"].mean() else COL_LEGIT
              for v in region_fraud["mean"].values]
    ax2.barh(labels2, region_fraud["mean"].values * 100, color=colors, edgecolor="white")
    ax2.axvline(df["is_fraud"].mean() * 100, color="gray", linestyle="--", alpha=0.7)
    ax2.set_title("Tasa de Fraude por Región (%)", fontweight="bold")
    ax2.set_xlabel("Tasa de Fraude (%)")
    
    plt.tight_layout()
    fig.savefig(EDA_DIR / "09_analisis_geografico.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Análisis geográfico")


def plot_summary_statistics(df):
    """10. Resumen estadístico visual"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Resumen Estadístico del Dataset de Fraude en E-commerce",
                 fontsize=16, fontweight="bold", y=1.02)
    
    # KPIs
    metrics = {
        "Total Transacciones": f"{len(df):,}",
        "Tasa de Fraude": f"{df['is_fraud'].mean()*100:.1f}%",
        "Monto Promedio": f"S/ {df['transaction_amount'].mean():,.0f}",
        "Monto Mediano": f"S/ {df['transaction_amount'].median():,.0f}",
        "Regiones Únicas": f"{df['customer_region'].nunique()}",
        "Bancos Únicos": f"{df['issuer_bank'].nunique()}",
    }
    
    for ax, (title, value) in zip(axes.flat, metrics.items()):
        ax.text(0.5, 0.6, value, fontsize=28, fontweight="bold",
                ha="center", va="center", color=COL_PRIMARY)
        ax.text(0.5, 0.25, title, fontsize=13, ha="center", va="center",
                alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False,
                                    edgecolor="#bdc3c7", linewidth=2,
                                    transform=ax.transAxes))
    
    plt.tight_layout()
    fig.savefig(EDA_DIR / "10_resumen_estadistico.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Resumen estadístico")

def export_eda_data(df):
    """Extrae la data de los gráficos a JSON para permitir personalización sin re-ejecutar."""
    eda_data = {}
    
    # 1. Distribución de clases
    eda_data["class_distribution"] = df["is_fraud"].value_counts().sort_index().tolist()
    
    # 3. Temporales
    eda_data["hourly_legit"] = df[df["is_fraud"]==0].groupby("hour").size().reindex(range(24), fill_value=0).tolist()
    eda_data["hourly_fraud"] = df[df["is_fraud"]==1].groupby("hour").size().reindex(range(24), fill_value=0).tolist()
    eda_data["fraud_by_day"] = (df.groupby("day_of_week")["is_fraud"].mean() * 100).fillna(0).tolist()
    eda_data["fraud_by_month"] = (df.groupby("month")["is_fraud"].mean() * 100).fillna(0).tolist()
    
    # 4. Categorías Top 8
    cats = ["card_brand", "card_type", "payment_channel", "issuer_bank", "customer_region", "category"]
    eda_data["categorical"] = {}
    for col in cats:
        top_cats = df[col].value_counts().head(8).index
        sub = df[df[col].isin(top_cats)]
        fraud_rate = sub.groupby(col)["is_fraud"].mean().sort_values(ascending=True) * 100
        eda_data["categorical"][col] = {
            "labels": fraud_rate.index.tolist(),
            "fraud_rate_pct": fraud_rate.tolist()
        }
        
    # 5. Tipos de Fraude
    fraud_df = df[df["is_fraud"] == 1]
    if len(fraud_df) > 0:
        type_counts = fraud_df["fraud_type"].value_counts()
        eda_data["fraud_types"] = {
            "labels": type_counts.index.tolist(),
            "counts": type_counts.tolist(),
            "avg_amount": fraud_df.groupby("fraud_type")["transaction_amount"].mean().reindex(type_counts.index).fillna(0).tolist()
        }
    
    # 6. Biometría Conductual (Para el paper)
    if "session_duration_minutes" in df.columns:
        # Sampleamos 500 puntos para gráficas de dispersión ligeras
        s_legit = min(500, len(df[df["is_fraud"]==0]))
        s_fraud = min(500, len(df[df["is_fraud"]==1]))
        eda_data["biometrics"] = {
            "legit_duration": df[df["is_fraud"]==0]["session_duration_minutes"].sample(s_legit).tolist(),
            "legit_velocity": df[df["is_fraud"]==0]["interaction_velocity"].sample(s_legit).tolist(),
            "fraud_duration": df[df["is_fraud"]==1]["session_duration_minutes"].sample(s_fraud).tolist(),
            "fraud_velocity": df[df["is_fraud"]==1]["interaction_velocity"].sample(s_fraud).tolist()
        }
    
    out_path = EDA_DIR / "eda_plot_data.json"
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eda_data, f, indent=2, ensure_ascii=False)
    print("  ✓ Datos JSON exportados a eda_plot_data.json")

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("ANÁLISIS EXPLORATORIO DE DATOS — Fraude en E-commerce Perú")
    print("=" * 70)
    
    df = load_data()
    
    print(f"\nGenerando gráficos en: {EDA_DIR}")
    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_temporal_patterns(df)
    plot_categorical_analysis(df)
    plot_fraud_types(df)
    plot_correlation_matrix(df)
    plot_eci_analysis(df)
    plot_behavioral_features(df)
    plot_geographic_analysis(df)
    plot_summary_statistics(df)
    
    export_eda_data(df)
    
    print(f"\n✅ 10 gráficos de EDA y JSON exportados en: {EDA_DIR}")
