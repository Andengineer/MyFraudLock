#!/usr/bin/env python3
"""
02_eda_analysis.py — Exploratory Data Analysis for the Niubiz/Socopur
fraud-detection dataset (post-rewrite for the current 26-column raw
schema with B2 correlations).

Generates 10 journal-quality figures in outputs/eda/.
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "outputs" / "data"
EDA_DIR  = BASE_DIR / "outputs" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# ── Style (matches 06b/regen for journal consistency) ─────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 10.5,
    'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 9.5,
    'legend.frameon': True, 'legend.edgecolor': '#CCCCCC', 'legend.fancybox': False,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 4, 'ytick.major.size': 4,
})

BLUE_DARK  = '#0D2137'; BLUE_MED   = '#1A4A7A'; BLUE_MAIN  = '#2166AC'
BLUE_LIGHT = '#74B0D4'; BLUE_PALE  = '#D1E5F0'
ACCENT     = '#B2182B'; ACCENT_MED = '#D6604D'; ACCENT_LIGHT= '#F4A58A'; ACCENT_PALE= '#FDDBC7'
GRID_C = '#E8E8E8'; TEXT_C = '#1A1A1A'; SUBTEXT_C = '#555555'
PAL = {"Legitimate": BLUE_MAIN, "Fraudulent": ACCENT}


def style_ax(ax, grid_axis='y'):
    ax.spines['left'].set_linewidth(0.7);   ax.spines['bottom'].set_linewidth(0.7)
    ax.spines['left'].set_color('#AAAAAA'); ax.spines['bottom'].set_color('#AAAAAA')
    ax.tick_params(colors=SUBTEXT_C, length=3)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID_C, linewidth=0.6, linestyle='--', alpha=0.8)
        ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(EDA_DIR / name, facecolor='white', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  ✓ {name}")


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING (raw dataset — has `is_fraud` and `fraud_type_meta`)
# ══════════════════════════════════════════════════════════════════════
def load_data():
    df = pd.read_csv(DATA_DIR / "fraud_ecommerce_dataset_raw.csv")
    print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
    # Normalize casing for categorical strings (raw data has injected format
    # inconsistencies — Lima/LIMA, ripley/Ripley — same as 02b preprocess does)
    cat_cols = ["card_brand", "card_type", "email_domain", "transaction_status",
                 "issuer_bank", "payment_channel", "wallet_yape", "wallet_plin",
                 "customer_region", "product_category"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
            df[c] = df[c].replace({"nan": None})
    # Derive temporal features for EDA (raw has datetime as string)
    df["transaction_datetime"] = pd.to_datetime(
        df["transaction_datetime"], errors="coerce", format="mixed")
    df["hour"]        = df["transaction_datetime"].dt.hour
    df["day_of_week"] = df["transaction_datetime"].dt.dayofweek
    df["month"]       = df["transaction_datetime"].dt.month
    df["label"]       = df["is_fraud"].map({0: "Legitimate", 1: "Fraudulent"})
    return df


# ══════════════════════════════════════════════════════════════════════
# 1. CLASS DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════
def plot_class_distribution(df):
    counts = df["is_fraud"].value_counts().sort_index().values
    total = counts.sum()
    pcts = counts / total * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), facecolor='white',
                                     gridspec_kw={'width_ratios': [1.3, 1], 'wspace': 0.25})

    # Bar
    bars = ax1.bar(["Legitimate", "Fraudulent"], counts,
                    color=[BLUE_MAIN, ACCENT], edgecolor='white', linewidth=1.2)
    # Add ~15% headroom so text doesn't collide with title
    ax1.set_ylim(0, max(counts) * 1.18)
    for bar, val, pct in zip(bars, counts, pcts):
        ax1.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + max(counts) * 0.02,
                  f"{int(val):,}\n({pct:.2f}%)", ha="center", fontsize=11,
                  color=TEXT_C, fontweight='bold')
    ax1.set_ylabel("Transaction count", fontweight='bold')
    ax1.set_title("Class count", fontweight='bold', color=BLUE_DARK, pad=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    style_ax(ax1)

    # Pie
    ax2.pie(counts, labels=["Legitimate", "Fraudulent"],
             colors=[BLUE_MAIN, ACCENT], autopct="%1.2f%%",
             startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2},
             textprops={'fontsize': 10.5, 'color': TEXT_C})
    ax2.set_title("Class proportion", fontweight='bold', color=BLUE_DARK, pad=10)

    fig.suptitle("Class Distribution — Niubiz/Socopur Dataset",
                 fontweight='bold', fontsize=13, color=BLUE_DARK, y=1.02)
    save(fig, "01_distribucion_clases.png")


# ══════════════════════════════════════════════════════════════════════
# 2. AMOUNT DISTRIBUTION (log scale)
# ══════════════════════════════════════════════════════════════════════
def plot_amount_distribution(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='white',
                                     gridspec_kw={'wspace': 0.25})
    legit = df[df["label"] == "Legitimate"]["transaction_amount"]
    fraud = df[df["label"] == "Fraudulent"]["transaction_amount"]

    # KDE log-amount
    log_legit = np.log1p(legit.clip(0.1, 5000))
    log_fraud = np.log1p(fraud.clip(0.1, 5000))
    sns.kdeplot(log_legit, ax=ax1, color=BLUE_MAIN, fill=True, alpha=0.4,
                  label=f"Legitimate (n={len(legit):,})", linewidth=2)
    sns.kdeplot(log_fraud, ax=ax1, color=ACCENT, fill=True, alpha=0.4,
                  label=f"Fraudulent (n={len(fraud):,})", linewidth=2)
    ax1.set_xlabel("log(1 + transaction_amount)", fontweight='bold')
    ax1.set_ylabel("Density", fontweight='bold')
    ax1.set_title("Amount density by class (log)", fontweight='bold', color=BLUE_DARK)
    # Add 18% headroom and place legend at upper-left, away from the curves
    cur_top = ax1.get_ylim()[1]
    ax1.set_ylim(0, cur_top * 1.18)
    ax1.legend(loc='upper left', framealpha=0.92, edgecolor='#CCCCCC')
    style_ax(ax1, grid_axis='both')

    # Boxplot (linear, with outliers capped)
    bp_data = [legit.clip(0, 2500), fraud.clip(0, 2500)]
    bp = ax2.boxplot(bp_data, labels=["Legitimate", "Fraudulent"],
                       patch_artist=True, widths=0.55)
    for patch, color in zip(bp['boxes'], [BLUE_MAIN, ACCENT]):
        patch.set_facecolor(color); patch.set_alpha(0.6); patch.set_edgecolor(color)
    for median in bp['medians']:
        median.set_color('black'); median.set_linewidth(1.5)
    ax2.set_ylabel("Transaction amount (PEN)", fontweight='bold')
    ax2.set_title("Amount boxplot (capped at S/.2,500)",
                   fontweight='bold', color=BLUE_DARK)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    style_ax(ax2)

    fig.suptitle("Transaction Amount — Distribution Comparison",
                 fontweight='bold', fontsize=13, color=BLUE_DARK, y=1.02)
    save(fig, "02_distribucion_montos.png")


# ══════════════════════════════════════════════════════════════════════
# 3. TEMPORAL PATTERNS
# ══════════════════════════════════════════════════════════════════════
def plot_temporal_patterns(df):
    fig = plt.figure(figsize=(14, 8), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.30)

    # Hour heatmap (fraud rate by day×hour)
    ax1 = fig.add_subplot(gs[0, :])
    pivot = (df.groupby(["day_of_week", "hour"])["is_fraud"].mean() * 100).unstack(fill_value=0)
    sns.heatmap(pivot, cmap="YlOrRd", annot=False, ax=ax1,
                  cbar_kws={"label": "Fraud rate (%)"})
    ax1.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)
    ax1.set_xlabel("Hour of day", fontweight='bold')
    ax1.set_ylabel("Day of week", fontweight='bold')
    ax1.set_title("Fraud rate heatmap (day × hour)",
                   fontweight='bold', color=BLUE_DARK)

    # Fraud rate by hour
    ax2 = fig.add_subplot(gs[1, 0])
    hourly = (df.groupby("hour")["is_fraud"].mean() * 100)
    ax2.plot(hourly.index, hourly.values, color=ACCENT, linewidth=2, marker='o', markersize=4)
    ax2.axhline(df["is_fraud"].mean() * 100, color=BLUE_MAIN, linestyle='--',
                 alpha=0.6, label=f"Mean rate {df['is_fraud'].mean()*100:.2f}%")
    ax2.set_xlabel("Hour of day", fontweight='bold')
    ax2.set_ylabel("Fraud rate (%)", fontweight='bold')
    ax2.set_title("Fraud rate by hour", fontweight='bold', color=BLUE_DARK)
    ax2.set_xticks(range(0, 24, 3))
    ax2.legend()
    style_ax(ax2, grid_axis='both')

    # Fraud rate by month
    ax3 = fig.add_subplot(gs[1, 1])
    monthly = (df.groupby("month")["is_fraud"].mean() * 100)
    ax3.bar(monthly.index, monthly.values, color=ACCENT_LIGHT,
             edgecolor=ACCENT, linewidth=1)
    ax3.axhline(df["is_fraud"].mean() * 100, color=BLUE_MAIN, linestyle='--', alpha=0.6)
    ax3.set_xlabel("Month", fontweight='bold')
    ax3.set_ylabel("Fraud rate (%)", fontweight='bold')
    ax3.set_title("Fraud rate by month", fontweight='bold', color=BLUE_DARK)
    style_ax(ax3, grid_axis='both')

    fig.suptitle("Temporal Patterns of Fraud Activity",
                 fontweight='bold', fontsize=14, color=BLUE_DARK, y=0.98)
    save(fig, "03_patrones_temporales.png")


# ══════════════════════════════════════════════════════════════════════
# 4. CATEGORICAL ANALYSIS (issuer_bank, card_brand, card_type)
# ══════════════════════════════════════════════════════════════════════
def plot_categorical_analysis(df):
    cats = ["card_brand", "card_type", "issuer_bank"]
    titles = ["Card brand", "Card type", "Issuer bank"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor='white',
                                gridspec_kw={'wspace': 0.32})
    overall = df["is_fraud"].mean() * 100
    for ax, col, title in zip(axes, cats, titles):
        sub = df[df[col].notna()]
        fr = (sub.groupby(col)["is_fraud"].mean() * 100).sort_values()
        colors = [ACCENT if v > overall else BLUE_LIGHT for v in fr.values]
        bars = ax.barh(range(len(fr)), fr.values, color=colors, edgecolor="white")
        for bar, val in zip(bars, fr.values):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}%", va='center', fontsize=8.5, color=TEXT_C)
        ax.set_yticks(range(len(fr)))
        ax.set_yticklabels(fr.index, fontsize=9)
        ax.axvline(overall, color=BLUE_DARK, linestyle='--', alpha=0.6,
                    label=f"Avg {overall:.2f}%")
        ax.set_xlabel("Fraud rate (%)", fontweight='bold')
        ax.set_title(title, fontweight='bold', color=BLUE_DARK)
        ax.legend(fontsize=8, loc='lower right')
        style_ax(ax, grid_axis='x')
    fig.suptitle("Fraud Rate by Categorical Feature",
                  fontweight='bold', fontsize=14, color=BLUE_DARK, y=1.04)
    save(fig, "04_analisis_categorico.png")


# ══════════════════════════════════════════════════════════════════════
# 5. FRAUD TYPES BREAKDOWN
# ══════════════════════════════════════════════════════════════════════
def plot_fraud_types(df):
    fraud_df = df[df["is_fraud"] == 1]
    if "fraud_type_meta" not in fraud_df.columns:
        print("  ⚠ fraud_type_meta not in dataset (skipping)")
        return
    type_counts = fraud_df["fraud_type_meta"].value_counts()
    pcts = type_counts / type_counts.sum() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor='white',
                                     gridspec_kw={'width_ratios': [1.3, 1], 'wspace': 0.18})

    # Bar chart
    colors_n = sns.color_palette("flare", n_colors=len(type_counts)).as_hex()
    bars = ax1.bar(range(len(type_counts)), type_counts.values,
                    color=colors_n, edgecolor='white')
    for bar, val, pct in zip(bars, type_counts.values, pcts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                  f"{int(val):,}\n({pct:.1f}%)", ha="center", fontsize=9.5,
                  color=TEXT_C, fontweight='bold')
    ax1.set_xticks(range(len(type_counts)))
    ax1.set_xticklabels(type_counts.index, rotation=20, ha='right')
    ax1.set_ylabel("Count", fontweight='bold')
    ax1.set_title("Fraud typology breakdown (count)",
                   fontweight='bold', color=BLUE_DARK)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    style_ax(ax1)

    # Avg amount per fraud type
    avg_amt = fraud_df.groupby("fraud_type_meta")["transaction_amount"].mean()
    avg_amt = avg_amt.reindex(type_counts.index)
    bars2 = ax2.barh(range(len(avg_amt)), avg_amt.values,
                       color=colors_n, edgecolor='white')
    for bar, val in zip(bars2, avg_amt.values):
        ax2.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                  f"S/. {val:,.0f}", va='center', fontsize=9, color=TEXT_C)
    ax2.set_yticks(range(len(avg_amt)))
    ax2.set_yticklabels(avg_amt.index, fontsize=9.5)
    ax2.set_xlabel("Mean transaction amount (PEN)", fontweight='bold')
    ax2.set_title("Avg amount per typology",
                   fontweight='bold', color=BLUE_DARK)
    style_ax(ax2, grid_axis='x')

    fig.suptitle("Fraud Typology Analysis (n = {:,} fraud transactions)".format(
        len(fraud_df)), fontweight='bold', fontsize=14, color=BLUE_DARK, y=1.02)
    save(fig, "05_tipos_fraude.png")


# ══════════════════════════════════════════════════════════════════════
# 6. CORRELATION MATRIX (numeric features)
# ══════════════════════════════════════════════════════════════════════
def plot_correlation_matrix(df):
    numeric_cols = ["transaction_amount", "discount_amount", "hour",
                      "day_of_week", "month", "eci", "action_code",
                      "num_installments", "num_items", "is_fraud"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[numeric_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                  center=0, vmin=-1, vmax=1, ax=ax, square=True,
                  cbar_kws={"label": "Pearson correlation",
                             "shrink": 0.7},
                  annot_kws={"size": 9})
    ax.set_title("Correlation Matrix — Numeric Features",
                  fontweight='bold', fontsize=13, color=BLUE_DARK, pad=12)
    save(fig, "06_correlacion.png")


# ══════════════════════════════════════════════════════════════════════
# 7. ECI / AUTHENTICATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════
def plot_eci_analysis(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor='white',
                                     gridspec_kw={'wspace': 0.30})

    # ECI distribution by class
    eci_rate = (df.groupby("eci")["is_fraud"].mean() * 100).sort_index()
    counts   = df["eci"].value_counts().sort_index()
    overall  = df["is_fraud"].mean() * 100
    eci_labels = {0: "0 (no-auth)", 2: "2 (MC-SC)", 5: "5 (VbV-3DS)",
                    6: "6 (3DS-fail)", 7: "7 (Yape/No3DS)", 11: "11 (other)"}
    x_labels = [eci_labels.get(int(i), str(int(i))) for i in eci_rate.index]
    colors_e = [ACCENT if v > overall else BLUE_LIGHT for v in eci_rate.values]
    bars = ax1.bar(range(len(eci_rate)), eci_rate.values,
                    color=colors_e, edgecolor='white')
    for bar, val, cnt in zip(bars, eci_rate.values, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                  f"{val:.1f}%\n(n={int(cnt):,})", ha='center', fontsize=8.5,
                  color=TEXT_C, fontweight='bold')
    ax1.set_xticks(range(len(eci_rate)))
    ax1.set_xticklabels(x_labels, rotation=20, ha='right', fontsize=8.5)
    ax1.axhline(overall, color=BLUE_DARK, linestyle='--', alpha=0.6,
                 label=f"Avg {overall:.2f}%")
    ax1.set_ylabel("Fraud rate (%)", fontweight='bold')
    ax1.set_title("Fraud rate by ECI code",
                   fontweight='bold', color=BLUE_DARK)
    ax1.legend(fontsize=8.5)
    style_ax(ax1)

    # Transaction status
    status_rate = (df.groupby("transaction_status")["is_fraud"].mean() * 100).sort_values()
    colors_s = [ACCENT if v > overall else BLUE_LIGHT for v in status_rate.values]
    bars2 = ax2.bar(range(len(status_rate)), status_rate.values,
                      color=colors_s, edgecolor='white')
    for bar, val in zip(bars2, status_rate.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                  f"{val:.1f}%", ha='center', fontsize=9.5,
                  color=TEXT_C, fontweight='bold')
    ax2.set_xticks(range(len(status_rate)))
    ax2.set_xticklabels(status_rate.index)
    ax2.axhline(overall, color=BLUE_DARK, linestyle='--', alpha=0.6)
    ax2.set_ylabel("Fraud rate (%)", fontweight='bold')
    ax2.set_title("Fraud rate by transaction status",
                   fontweight='bold', color=BLUE_DARK)
    style_ax(ax2)

    fig.suptitle("Authentication & Transaction Status Analysis",
                  fontweight='bold', fontsize=13, color=BLUE_DARK, y=1.04)
    save(fig, "07_analisis_eci.png")


# ══════════════════════════════════════════════════════════════════════
# 8. BEHAVIORAL FEATURES (wallet usage, num_installments, num_items)
# ══════════════════════════════════════════════════════════════════════
def plot_behavioral_features(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor='white',
                                gridspec_kw={'wspace': 0.30, 'hspace': 0.45})
    overall = df["is_fraud"].mean() * 100

    # Yape usage
    ax = axes[0, 0]
    yape = (df.groupby("wallet_yape")["is_fraud"].mean() * 100)
    bars = ax.bar(yape.index, yape.values,
                   color=[BLUE_LIGHT if v < overall else ACCENT for v in yape.values],
                   edgecolor='white')
    for bar, val in zip(bars, yape.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{val:.2f}%", ha='center', fontsize=10, color=TEXT_C, fontweight='bold')
    ax.axhline(overall, color=BLUE_DARK, linestyle='--', alpha=0.6)
    ax.set_title("Fraud rate by Yape usage",
                  fontweight='bold', color=BLUE_DARK)
    ax.set_ylabel("Fraud rate (%)", fontweight='bold')
    style_ax(ax)

    # Plin usage
    ax = axes[0, 1]
    plin = (df.groupby("wallet_plin")["is_fraud"].mean() * 100)
    bars = ax.bar(plin.index, plin.values,
                   color=[BLUE_LIGHT if v < overall else ACCENT for v in plin.values],
                   edgecolor='white')
    for bar, val in zip(bars, plin.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{val:.2f}%", ha='center', fontsize=10, color=TEXT_C, fontweight='bold')
    ax.axhline(overall, color=BLUE_DARK, linestyle='--', alpha=0.6)
    ax.set_title("Fraud rate by Plin usage",
                  fontweight='bold', color=BLUE_DARK)
    ax.set_ylabel("Fraud rate (%)", fontweight='bold')
    style_ax(ax)

    # Num installments
    ax = axes[1, 0]
    inst = (df.groupby("num_installments")["is_fraud"].mean() * 100)
    ax.bar(inst.index.astype(str), inst.values, color=BLUE_MED, edgecolor='white')
    ax.axhline(overall, color=ACCENT, linestyle='--', alpha=0.7,
                label=f"Avg {overall:.2f}%")
    ax.set_xlabel("Installments", fontweight='bold')
    ax.set_ylabel("Fraud rate (%)", fontweight='bold')
    ax.set_title("Fraud rate by installments",
                  fontweight='bold', color=BLUE_DARK)
    ax.legend(fontsize=8.5)
    style_ax(ax)

    # Payment channel
    ax = axes[1, 1]
    pc = (df.groupby("payment_channel")["is_fraud"].mean() * 100).sort_values()
    bars = ax.barh(range(len(pc)), pc.values,
                    color=[BLUE_LIGHT if v < overall else ACCENT for v in pc.values],
                    edgecolor='white')
    for bar, val in zip(bars, pc.values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}%", va='center', fontsize=10, color=TEXT_C, fontweight='bold')
    ax.set_yticks(range(len(pc)))
    ax.set_yticklabels(pc.index, fontsize=9.5)
    ax.axvline(overall, color=BLUE_DARK, linestyle='--', alpha=0.6)
    ax.set_xlabel("Fraud rate (%)", fontweight='bold')
    ax.set_title("Fraud rate by payment channel",
                  fontweight='bold', color=BLUE_DARK)
    style_ax(ax, grid_axis='x')

    fig.suptitle("Behavioral Features Analysis",
                  fontweight='bold', fontsize=14, color=BLUE_DARK, y=0.99)
    save(fig, "08_features_comportamentales.png")


# ══════════════════════════════════════════════════════════════════════
# 9. GEOGRAPHIC ANALYSIS (customer_region)
# ══════════════════════════════════════════════════════════════════════
def plot_geographic_analysis(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), facecolor='white',
                                     gridspec_kw={'wspace': 0.45})
    overall = df["is_fraud"].mean() * 100

    # Top regions by count + fraud rate
    region_data = df.groupby("customer_region").agg(
        n=("is_fraud", "size"),
        fraud_rate=("is_fraud", lambda x: x.mean() * 100),
    ).sort_values("n", ascending=False).head(15)

    # Counts
    bars1 = ax1.barh(range(len(region_data)), region_data["n"].values,
                       color=BLUE_LIGHT, edgecolor='white')
    for bar, val in zip(bars1, region_data["n"].values):
        ax1.text(bar.get_width() + max(region_data["n"]) * 0.01,
                  bar.get_y() + bar.get_height()/2,
                  f"{int(val):,}", va='center', fontsize=9, color=TEXT_C)
    ax1.set_yticks(range(len(region_data)))
    ax1.set_yticklabels(region_data.index, fontsize=9.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Transaction count", fontweight='bold')
    ax1.set_title("Top 15 regions by volume",
                   fontweight='bold', color=BLUE_DARK)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    style_ax(ax1, grid_axis='x')

    # Fraud rates (same regions)
    colors_g = [ACCENT if v > overall else BLUE_LIGHT for v in region_data["fraud_rate"].values]
    bars2 = ax2.barh(range(len(region_data)), region_data["fraud_rate"].values,
                       color=colors_g, edgecolor='white')
    for bar, val in zip(bars2, region_data["fraud_rate"].values):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                  f"{val:.2f}%", va='center', fontsize=9, color=TEXT_C, fontweight='bold')
    ax2.set_yticks(range(len(region_data)))
    ax2.set_yticklabels(region_data.index, fontsize=9.5)
    ax2.invert_yaxis()
    ax2.axvline(overall, color=BLUE_DARK, linestyle='--', alpha=0.6,
                  label=f"Avg {overall:.2f}%")
    ax2.set_xlabel("Fraud rate (%)", fontweight='bold')
    ax2.set_title("Fraud rate by region",
                   fontweight='bold', color=BLUE_DARK)
    ax2.legend(fontsize=8.5)
    style_ax(ax2, grid_axis='x')

    fig.suptitle("Geographic Analysis — Peruvian Regions",
                  fontweight='bold', fontsize=14, color=BLUE_DARK, y=0.99)
    save(fig, "09_analisis_geografico.png")


# ══════════════════════════════════════════════════════════════════════
# 10. STATISTICAL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════
def plot_summary_table(df):
    legit = df[df["is_fraud"] == 0]
    fraud = df[df["is_fraud"] == 1]

    summary = pd.DataFrame({
        "Variable": ["transaction_amount", "hour", "num_installments", "num_items",
                       "eci"],
        "Legit Mean": [
            f"{legit['transaction_amount'].mean():,.2f}",
            f"{legit['hour'].mean():.1f}",
            f"{legit['num_installments'].mean():.2f}",
            f"{legit['num_items'].mean():.2f}",
            f"{legit['eci'].mean():.2f}",
        ],
        "Legit Std":  [
            f"{legit['transaction_amount'].std():,.2f}",
            f"{legit['hour'].std():.2f}",
            f"{legit['num_installments'].std():.2f}",
            f"{legit['num_items'].std():.2f}",
            f"{legit['eci'].std():.2f}",
        ],
        "Fraud Mean": [
            f"{fraud['transaction_amount'].mean():,.2f}",
            f"{fraud['hour'].mean():.1f}",
            f"{fraud['num_installments'].mean():.2f}",
            f"{fraud['num_items'].mean():.2f}",
            f"{fraud['eci'].mean():.2f}",
        ],
        "Fraud Std":  [
            f"{fraud['transaction_amount'].std():,.2f}",
            f"{fraud['hour'].std():.2f}",
            f"{fraud['num_installments'].std():.2f}",
            f"{fraud['num_items'].std():.2f}",
            f"{fraud['eci'].std():.2f}",
        ],
    })

    fig, ax = plt.subplots(figsize=(11, 4), facecolor='white')
    ax.axis("off")
    table = ax.table(cellText=summary.values, colLabels=summary.columns,
                       cellLoc="center", loc="center",
                       colColours=[BLUE_MED] * len(summary.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.8)
    for j in range(len(summary.columns)):
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Statistical Summary — Numeric Features by Class",
                  fontweight='bold', fontsize=13, color=BLUE_DARK, pad=20)
    save(fig, "10_resumen_estadistico.png")

    # Also save raw stats to JSON for paper
    stats_export = {
        "n_total":  int(len(df)),
        "n_legit":  int(len(legit)),
        "n_fraud":  int(len(fraud)),
        "fraud_rate": float(df["is_fraud"].mean()),
        "fraud_typology": dict(fraud["fraud_type_meta"].value_counts())
            if "fraud_type_meta" in df.columns else {},
        "amount_stats": {
            "legit": {"mean": float(legit["transaction_amount"].mean()),
                       "median": float(legit["transaction_amount"].median()),
                       "p95": float(legit["transaction_amount"].quantile(0.95))},
            "fraud": {"mean": float(fraud["transaction_amount"].mean()),
                       "median": float(fraud["transaction_amount"].median()),
                       "p95": float(fraud["transaction_amount"].quantile(0.95))},
        },
    }
    with open(EDA_DIR / "eda_plot_data.json", "w") as f:
        json.dump(stats_export, f, indent=2, default=str)
    print("  ✓ eda_plot_data.json")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("EDA — Niubiz/Socopur Fraud Dataset (new schema)")
    print("=" * 70)
    df = load_data()
    print()
    print("Generating figures…")
    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_temporal_patterns(df)
    plot_categorical_analysis(df)
    plot_fraud_types(df)
    plot_correlation_matrix(df)
    plot_eci_analysis(df)
    plot_behavioral_features(df)
    plot_geographic_analysis(df)
    plot_summary_table(df)
    print("\n✅ EDA completado — 10 figuras en outputs/eda/")


if __name__ == "__main__":
    main()
