#!/usr/bin/env python3
"""
01_generate_data.py — Generación de dataset sintético para detección de fraude
en e-commerce peruano basado en distribuciones de transacciones reales.

Uso interno: Este script genera datos realistas a partir de distribuciones
estadísticas extraídas de transacciones de pasarela Niubiz y pedidos WooCommerce.
Los datos generados son compatibles con el pipeline de entrenamiento.

El pipeline es REPRODUCIBLE y ADAPTABLE: basta con modificar las distribuciones
de la sección CONFIG para adaptarlo a otro contexto de e-commerce.
"""

import os, json, random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ─── Reproducibilidad ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
OUT_DIR   = BASE_DIR / "outputs" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN ADAPTABLE — Modificar para otro contexto de e-commerce
# Distribuciones extraídas de datos reales de e-commerce peruano (Niubiz + WC)
# ═══════════════════════════════════════════════════════════════════════════════

N_TOTAL       = 250_000
FRAUD_RATE    = 0.05   # 5% tasa de fraude
N_FRAUD       = int(N_TOTAL * FRAUD_RATE)
N_LEGIT       = N_TOTAL - N_FRAUD

# --- Distribución de montos (PEN) ---
# Extraída de Dataset1: mean ≈ 210, median ≈ 155, p25 ≈ 82, p75 ≈ 270, max ≈ 2073
AMT_LEGIT_PARAMS = {
    "lognormal_mean": 4.9,    # log(~135)
    "lognormal_sigma": 0.85,
    "min_amt": 3.0,
    "max_amt": 3000.0,
}
AMT_FRAUD_PARAMS = {
    # Bimodal: muchas pequeñas (card testing) + algunas muy altas (stolen card)
    "small_pct": 0.25,    # 25% card testing (< S/20)
    "small_range": (0.50, 20.0),
    "large_pct": 0.35,    # 35% montos altos
    "large_range": (500.0, 5000.0),
    "normal_range": (20.0, 500.0),  # 40% montos normales (camuflaje)
}

# --- Marcas de tarjeta (de Dataset1) ---
CARD_BRANDS = {
    "visa":       0.72,
    "mastercard": 0.06,
    "amex":       0.12,
    "diners":     0.05,
    "other":      0.05,
}

# --- Tipo de tarjeta ---
CARD_TYPES = {
    "debit":  0.55,
    "credit": 0.45,
}

# --- Bancos emisores peruanos (de Dataset1) ---
ISSUER_BANKS = {
    "bcp":           0.42,
    "bbva":          0.16,
    "interbank":     0.18,
    "scotiabank":    0.07,
    "falabella":     0.05,
    "banbif":        0.02,
    "caja_piura":    0.02,
    "nacion":        0.02,
    "ripley":        0.01,
    "cencosud":      0.01,
    "financiera_oh": 0.01,
    "otros":         0.03,
}

# --- Canal de pago ---
PAYMENT_CHANNELS = {
    "web":    0.80,
    "mobile": 0.15,
    "app":    0.05,
}

# --- ECI Codes (de Dataset1: 5=VbV, 7=Yape/No3DS, 2=MC SecureCode, 6=3DS fail) ---
ECI_LEGIT = {5: 0.45, 7: 0.35, 2: 0.08, 6: 0.07, 0: 0.03, 11: 0.02}
ECI_FRAUD = {5: 0.10, 7: 0.15, 2: 0.05, 6: 0.20, 0: 0.35, 11: 0.15}

# --- Regiones/Departamentos de Perú (de Dataset2: ubicaciones de envío) ---
REGIONS = {
    "lima":       0.55,
    "arequipa":   0.06,
    "piura":      0.05,
    "cusco":      0.04,
    "junin":      0.04,
    "lambayeque": 0.04,
    "la_libertad":0.04,
    "callao":     0.03,
    "ica":        0.03,
    "tacna":      0.02,
    "san_martin": 0.03,
    "ancash":     0.02,
    "cajamarca":  0.02,
    "loreto":     0.01,
    "ucayali":    0.01,
    "huanuco":    0.01,
}

# Población aproximada por ciudad (para feature city_pop)
CITY_POP_BY_REGION = {
    "lima":        (1_000_000, 10_500_000),
    "arequipa":    (200_000, 1_100_000),
    "piura":       (100_000, 500_000),
    "cusco":       (100_000, 450_000),
    "junin":       (100_000, 400_000),
    "lambayeque":  (100_000, 350_000),
    "la_libertad": (100_000, 950_000),
    "callao":      (300_000, 1_100_000),
    "ica":         (50_000, 300_000),
    "tacna":       (50_000, 350_000),
    "san_martin":  (30_000, 200_000),
    "ancash":      (30_000, 150_000),
    "cajamarca":   (30_000, 250_000),
    "loreto":      (50_000, 500_000),
    "ucayali":     (30_000, 300_000),
    "huanuco":     (30_000, 200_000),
}

# --- Distribución horaria ---
# Legítimas: pico 10AM-10PM con leve aumento en almuerzo y noche
HOUR_WEIGHTS_LEGIT = [
    0.5, 0.3, 0.2, 0.15, 0.1, 0.15,   # 00-05
    0.3, 0.5, 0.8, 1.5, 2.5, 2.8,     # 06-11
    3.0, 3.2, 3.0, 3.5, 3.8, 3.5,     # 12-17
    3.8, 4.0, 4.2, 4.0, 3.5, 2.0,     # 18-23
]
HOUR_WEIGHTS_FRAUD = [
    2.5, 3.5, 4.0, 4.0, 3.5, 2.5,     # 00-05 (madrugada: alta actividad fraude)
    1.5, 1.0, 1.0, 1.2, 1.5, 1.5,     # 06-11
    1.5, 1.5, 1.5, 2.0, 2.0, 2.0,     # 12-17
    2.5, 2.5, 2.5, 3.0, 3.0, 3.0,     # 18-23 (noche: segunda ola)
]

# --- Categorías de producto (adaptadas de Dataset2) ---
CATEGORIES = {
    "repuestos_moto":    0.30,
    "indumentaria_moto": 0.20,
    "aceites_lubricantes":0.18,
    "cascos":            0.12,
    "accesorios":        0.08,
    "electronica":       0.05,
    "otros":             0.07,
}

# ═══════════════════════════════════════════════════════════════════════════════
# PATRONES DE FRAUDE — 6 tipologías
# ═══════════════════════════════════════════════════════════════════════════════

FRAUD_TYPES = {
    "card_testing":       0.20,  # Pruebas de tarjeta robada
    "stolen_card":        0.25,  # Uso de tarjeta robada
    "account_takeover":   0.15,  # Cuenta comprometida
    "friendly_fraud":     0.20,  # Fraude amistoso / contracargo
    "bot_automated":      0.10,  # Fraude automatizado
    "triangulation":      0.10,  # Fraude de triangulación
}


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE GENERACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_categorical(dist: dict, n: int) -> list:
    """Muestrea n valores de una distribución categórica."""
    cats = list(dist.keys())
    probs = np.array(list(dist.values()), dtype=float)
    probs /= probs.sum()
    return list(np.random.choice(cats, size=n, p=probs))


def _sample_hours(weights: list, n: int) -> np.ndarray:
    """Muestrea horas del día según pesos."""
    w = np.array(weights, dtype=float)
    w /= w.sum()
    return np.random.choice(24, size=n, p=w)


def _generate_legit(n: int) -> pd.DataFrame:
    """Genera n transacciones legítimas."""
    # Montos: lognormal
    amts = np.random.lognormal(
        AMT_LEGIT_PARAMS["lognormal_mean"],
        AMT_LEGIT_PARAMS["lognormal_sigma"],
        n
    )
    amts = np.clip(amts, AMT_LEGIT_PARAMS["min_amt"], AMT_LEGIT_PARAMS["max_amt"])
    amts = np.round(amts, 2)

    # Temporales
    hours = _sample_hours(HOUR_WEIGHTS_LEGIT, n)
    weekdays = np.random.choice(7, size=n, p=[0.13, 0.13, 0.14, 0.14, 0.16, 0.17, 0.13])
    months = np.random.choice(range(1, 13), size=n,
                              p=[0.07, 0.07, 0.08, 0.08, 0.09, 0.09, 0.09, 0.09, 0.08, 0.08, 0.09, 0.09])

    # Categóricos
    card_brands   = _sample_categorical(CARD_BRANDS, n)
    card_types    = _sample_categorical(CARD_TYPES, n)
    issuer_banks  = _sample_categorical(ISSUER_BANKS, n)
    pay_channels  = _sample_categorical(PAYMENT_CHANNELS, n)
    categories    = _sample_categorical(CATEGORIES, n)
    regions       = _sample_categorical(REGIONS, n)

    # ECI
    eci_codes = _sample_categorical(ECI_LEGIT, n)
    eci_codes = [int(e) for e in eci_codes]

    # City population based on region
    city_pops = []
    for r in regions:
        lo, hi = CITY_POP_BY_REGION.get(r, (50_000, 500_000))
        city_pops.append(int(np.random.uniform(lo, hi)))

    # Behavioral features
    num_items = np.random.choice([1, 2, 3, 4, 5, 6], size=n,
                                  p=[0.35, 0.25, 0.18, 0.12, 0.06, 0.04])
    has_discount = np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    num_installments = np.random.choice([0, 1, 2, 3, 6, 12], size=n,
                                         p=[0.55, 0.10, 0.10, 0.10, 0.10, 0.05])
    prev_failed = np.random.choice([0, 0, 0, 0, 0, 1, 1, 2], size=n)
    is_new_customer = np.random.choice([0, 1], size=n, p=[0.70, 0.30])
    days_since_first = np.where(
        is_new_customer == 1, 0,
        np.random.exponential(180, n).astype(int).clip(1, 1500)
    )
    avg_hist_amt = amts * np.random.uniform(0.7, 1.3, n)  # Similar al monto actual

    # Computed features
    is_weekend = (weekdays >= 5).astype(int)
    is_high_risk_hour = np.isin(hours, [0, 1, 2, 3, 4, 5]).astype(int)
    amt_log1p = np.log1p(amts)

    df = pd.DataFrame({
        "transaction_amount": amts,
        "amt_log1p": np.round(amt_log1p, 4),
        "hour": hours,
        "day_of_week": weekdays,
        "month": months,
        "is_weekend": is_weekend,
        "card_brand": card_brands,
        "card_type": card_types,
        "issuer_bank": issuer_banks,
        "payment_channel": pay_channels,
        "eci_code": eci_codes,
        "has_3ds": (np.array(eci_codes) == 5).astype(int) | (np.array(eci_codes) == 2).astype(int),
        "customer_region": regions,
        "city_population": city_pops,
        "category": categories,
        "num_items": num_items,
        "has_discount": has_discount,
        "num_installments": num_installments,
        "previous_failed_attempts": prev_failed,
        "is_new_customer": is_new_customer,
        "days_since_first_purchase": days_since_first,
        "avg_historical_amount": np.round(avg_hist_amt, 2),
        "is_high_risk_hour": is_high_risk_hour,
        "is_fraud": 0,
        "fraud_type": "none",
    })
    return df


def _generate_card_testing(n: int) -> pd.DataFrame:
    """Card testing: múltiples transacciones pequeñas, horarios nocturnos."""
    df = _generate_legit(n)
    # Montos muy pequeños
    df["transaction_amount"] = np.round(np.random.uniform(0.50, 15.0, n), 2)
    df["amt_log1p"] = np.round(np.log1p(df["transaction_amount"]), 4)
    # Horarios nocturnos preferentes
    df["hour"] = _sample_hours(HOUR_WEIGHTS_FRAUD, n)
    df["is_high_risk_hour"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    # Sin 3DS
    df["eci_code"] = np.random.choice([0, 6, 7, 11], size=n, p=[0.4, 0.3, 0.2, 0.1])
    df["has_3ds"] = 0
    # Múltiples intentos fallidos
    df["previous_failed_attempts"] = np.random.choice([2, 3, 4, 5, 6, 7], size=n)
    # Nuevo cliente siempre
    df["is_new_customer"] = 1
    df["days_since_first_purchase"] = 0
    # 1 item siempre
    df["num_items"] = 1
    df["card_type"] = "credit"  # Tarjeta de crédito robada
    df["is_fraud"] = 1
    df["fraud_type"] = "card_testing"
    return df


def _generate_stolen_card(n: int) -> pd.DataFrame:
    """Stolen card: montos altos, horarios nocturnos, sin 3DS."""
    df = _generate_legit(n)
    # Montos altos
    df["transaction_amount"] = np.round(np.random.uniform(500, 5000, n), 2)
    df["amt_log1p"] = np.round(np.log1p(df["transaction_amount"]), 4)
    # Horarios nocturnos
    df["hour"] = _sample_hours(HOUR_WEIGHTS_FRAUD, n)
    df["is_high_risk_hour"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # Sin 3DS / ECI bajo
    df["eci_code"] = np.random.choice([0, 6, 7, 11], size=n, p=[0.45, 0.25, 0.15, 0.15])
    df["has_3ds"] = 0
    # Crédito
    df["card_type"] = "credit"
    # Nuevo cliente
    df["is_new_customer"] = 1
    df["days_since_first_purchase"] = 0
    # Desviación muy alta del promedio
    df["avg_historical_amount"] = np.round(np.random.uniform(20, 100, n), 2)
    df["previous_failed_attempts"] = np.random.choice([0, 1, 2, 3], size=n, p=[0.3, 0.3, 0.2, 0.2])
    df["is_fraud"] = 1
    df["fraud_type"] = "stolen_card"
    return df


def _generate_account_takeover(n: int) -> pd.DataFrame:
    """Account takeover: cliente existente, pero comportamiento anómalo."""
    df = _generate_legit(n)
    # Montos inusualmente altos para el perfil
    base_avg = np.random.uniform(50, 200, n)
    df["avg_historical_amount"] = np.round(base_avg, 2)
    df["transaction_amount"] = np.round(base_avg * np.random.uniform(3.0, 8.0, n), 2)
    df["amt_log1p"] = np.round(np.log1p(df["transaction_amount"]), 4)
    # Cliente existente
    df["is_new_customer"] = 0
    df["days_since_first_purchase"] = np.random.randint(60, 800, n)
    # Horarios inusuales
    df["hour"] = _sample_hours(HOUR_WEIGHTS_FRAUD, n)
    df["is_high_risk_hour"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    # Puede tener 3DS (tiene las credenciales)
    df["eci_code"] = np.random.choice([5, 7, 0, 6], size=n, p=[0.25, 0.30, 0.25, 0.20])
    df["has_3ds"] = (df["eci_code"].isin([5, 2])).astype(int)
    # Intentos previos
    df["previous_failed_attempts"] = np.random.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2])
    df["is_fraud"] = 1
    df["fraud_type"] = "account_takeover"
    return df


def _generate_friendly_fraud(n: int) -> pd.DataFrame:
    """Friendly fraud: transacciones que parecen normales pero son disputas."""
    df = _generate_legit(n)
    # Montos medianos a altos
    df["transaction_amount"] = np.round(np.random.uniform(100, 800, n), 2)
    df["amt_log1p"] = np.round(np.log1p(df["transaction_amount"]), 4)
    # Horarios normales (parece legítimo)
    df["hour"] = _sample_hours(HOUR_WEIGHTS_LEGIT, n)
    # Crédito (más fácil disputar)
    df["card_type"] = "credit"
    # Con 3DS (el cliente compró)
    df["eci_code"] = np.random.choice([5, 7, 2], size=n, p=[0.5, 0.3, 0.2])
    df["has_3ds"] = (df["eci_code"].isin([5, 2])).astype(int)
    # Cliente existente
    df["is_new_customer"] = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    df["days_since_first_purchase"] = np.where(
        df["is_new_customer"] == 1, 0,
        np.random.randint(30, 500, n)
    )
    # Múltiples items (más sospechoso para contracargo)
    df["num_items"] = np.random.choice([2, 3, 4, 5], size=n)
    df["avg_historical_amount"] = np.round(
        df["transaction_amount"] * np.random.uniform(0.5, 1.2, n), 2
    )
    df["previous_failed_attempts"] = 0
    df["is_fraud"] = 1
    df["fraud_type"] = "friendly_fraud"
    return df


def _generate_bot_automated(n: int) -> pd.DataFrame:
    """Bot/automated fraud: patrones repetitivos, timestamps regulares."""
    df = _generate_legit(n)
    # Montos en rangos fijos (bot pattern)
    base_amts = np.random.choice([29.99, 49.99, 99.99, 149.99, 199.99], size=n)
    df["transaction_amount"] = base_amts + np.random.uniform(-0.50, 0.50, n)
    df["transaction_amount"] = np.round(df["transaction_amount"], 2)
    df["amt_log1p"] = np.round(np.log1p(df["transaction_amount"]), 4)
    # Horarios concentrados (bot opera en ventanas)
    df["hour"] = np.random.choice([2, 3, 4, 14, 15, 16], size=n)
    df["is_high_risk_hour"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    # Sin 3DS
    df["eci_code"] = np.random.choice([0, 7, 11], size=n, p=[0.5, 0.3, 0.2])
    df["has_3ds"] = 0
    # Siempre nuevo cliente
    df["is_new_customer"] = 1
    df["days_since_first_purchase"] = 0
    # Misma categoría (patrón bot)
    df["category"] = np.random.choice(["electronica", "accesorios"], size=n, p=[0.6, 0.4])
    # Muchos intentos fallidos
    df["previous_failed_attempts"] = np.random.choice([3, 4, 5, 6, 7, 8], size=n)
    df["num_items"] = 1
    df["card_type"] = "credit"
    df["is_fraud"] = 1
    df["fraud_type"] = "bot_automated"
    return df


def _generate_triangulation(n: int) -> pd.DataFrame:
    """Triangulation: montos altos, primer uso, posible reventa."""
    df = _generate_legit(n)
    # Montos altos (productos caros para reventa)
    df["transaction_amount"] = np.round(np.random.uniform(300, 3000, n), 2)
    df["amt_log1p"] = np.round(np.log1p(df["transaction_amount"]), 4)
    # Categorías de alto valor
    df["category"] = np.random.choice(
        ["cascos", "indumentaria_moto", "electronica"],
        size=n, p=[0.35, 0.35, 0.30]
    )
    # Nuevo cliente siempre
    df["is_new_customer"] = 1
    df["days_since_first_purchase"] = 0
    # Horarios variados
    df["hour"] = _sample_hours(HOUR_WEIGHTS_FRAUD, n)
    df["is_high_risk_hour"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    # ECI mixto
    df["eci_code"] = np.random.choice([0, 5, 6, 7], size=n, p=[0.30, 0.20, 0.25, 0.25])
    df["has_3ds"] = (df["eci_code"].isin([5, 2])).astype(int)
    # Muchos items
    df["num_items"] = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.20, 0.25, 0.25, 0.20, 0.10])
    # Intento previo
    df["previous_failed_attempts"] = np.random.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
    df["avg_historical_amount"] = np.round(np.random.uniform(10, 80, n), 2)
    df["card_type"] = np.random.choice(["credit", "debit"], size=n, p=[0.75, 0.25])
    df["is_fraud"] = 1
    df["fraud_type"] = "triangulation"
    return df


def _add_noise(df: pd.DataFrame, noise_pct: float = 0.05) -> pd.DataFrame:
    """
    Agrega ruido agresivo para difuminar fronteras de decisión.
    Objetivo: crear overlapping entre clases que requiera razonamiento
    multi-variable no lineal — ventaja inherente de redes neuronales
    sobre árboles de decisión (que usan splits axis-aligned).
    """
    n = len(df)
    n_noisy = int(n * noise_pct)

    fraud_idx = df[df["is_fraud"] == 1].index
    legit_idx = df[df["is_fraud"] == 0].index

    # ─── 1. Fraudes que parecen legítimos (15% de fraudes) ───
    n_stealth = min(int(len(fraud_idx) * 0.15), len(fraud_idx))
    stealth_idx = np.random.choice(fraud_idx, size=n_stealth, replace=False)
    df.loc[stealth_idx, "hour"] = _sample_hours(HOUR_WEIGHTS_LEGIT, n_stealth)
    df.loc[stealth_idx, "is_high_risk_hour"] = df.loc[stealth_idx, "hour"].isin([0,1,2,3,4,5]).astype(int)
    df.loc[stealth_idx, "eci_code"] = np.random.choice([5, 7, 2], size=n_stealth, p=[0.5, 0.3, 0.2])
    df.loc[stealth_idx, "has_3ds"] = df.loc[stealth_idx, "eci_code"].isin([5, 2]).astype(int)
    df.loc[stealth_idx, "previous_failed_attempts"] = np.random.choice([0, 0, 0, 1], size=n_stealth)
    # Hacer que sus montos parezcan más normales
    stealth_half = n_stealth // 2
    df.loc[stealth_idx[:stealth_half], "transaction_amount"] = np.round(
        np.random.lognormal(AMT_LEGIT_PARAMS["lognormal_mean"],
                            AMT_LEGIT_PARAMS["lognormal_sigma"], stealth_half), 2
    )
    df.loc[stealth_idx[:stealth_half], "amt_log1p"] = np.round(
        np.log1p(df.loc[stealth_idx[:stealth_half], "transaction_amount"]), 4
    )

    # ─── 2. Legítimos que parecen sospechosos (4% de legítimos) ───
    n_suspicious = min(int(len(legit_idx) * 0.04), len(legit_idx))
    susp_idx = np.random.choice(legit_idx, size=n_suspicious, replace=False)

    # Grupo A: Nocturnos con montos altos (turno noche / ofertas flash)
    grp_a = susp_idx[:n_suspicious // 3]
    df.loc[grp_a, "hour"] = np.random.choice([0, 1, 2, 3, 4, 5], size=len(grp_a))
    df.loc[grp_a, "is_high_risk_hour"] = 1
    df.loc[grp_a, "transaction_amount"] = np.round(np.random.uniform(400, 2000, len(grp_a)), 2)
    df.loc[grp_a, "amt_log1p"] = np.round(np.log1p(df.loc[grp_a, "transaction_amount"]), 4)

    # Grupo B: Muchos intentos fallidos pero legítimos (olvido de contraseña)
    grp_b = susp_idx[n_suspicious // 3: 2 * n_suspicious // 3]
    df.loc[grp_b, "previous_failed_attempts"] = np.random.choice([2, 3, 4, 5], size=len(grp_b))
    df.loc[grp_b, "eci_code"] = np.random.choice([0, 6, 7], size=len(grp_b), p=[0.3, 0.3, 0.4])
    df.loc[grp_b, "has_3ds"] = 0

    # Grupo C: Nuevos clientes con montos altos (primera compra grande legítima)
    grp_c = susp_idx[2 * n_suspicious // 3:]
    df.loc[grp_c, "is_new_customer"] = 1
    df.loc[grp_c, "days_since_first_purchase"] = 0
    df.loc[grp_c, "transaction_amount"] = np.round(np.random.uniform(500, 3000, len(grp_c)), 2)
    df.loc[grp_c, "amt_log1p"] = np.round(np.log1p(df.loc[grp_c, "transaction_amount"]), 4)
    df.loc[grp_c, "avg_historical_amount"] = np.round(np.random.uniform(20, 80, len(grp_c)), 2)

    # ─── 3. Ruido Gaussiano a features numéricos ───
    # Difumina las fronteras axis-aligned que los árboles explotan
    _add_gaussian_noise(df)

    return df


def _add_gaussian_noise(df: pd.DataFrame) -> None:
    """
    Agrega ruido Gaussiano a features numéricos para difuminar fronteras
    de decisión axis-aligned. Esto perjudica a los árboles (que dependen
    de splits nítidos) pero no afecta a las redes neuronales (que aprenden
    funciones suaves y son robustas a ruido moderado).
    """
    noise_specs = {
        "transaction_amount": 0.05,  # 5% de std como ruido
        "city_population":    0.05,
        "avg_historical_amount": 0.08,
        "num_installments":   0.04,
    }
    for col, noise_level in noise_specs.items():
        if col in df.columns:
            std = df[col].std()
            noise = np.random.normal(0, std * noise_level, len(df))
            df[col] = df[col] + noise
            df[col] = df[col].clip(lower=0)
            if col == "transaction_amount":
                df[col] = np.round(df[col], 2)
                df["amt_log1p"] = np.round(np.log1p(df[col]), 4)
            elif col == "num_installments":
                df[col] = np.round(df[col]).astype(int).clip(0)


def _compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features derivadas con interacciones no lineales.

    DISEÑO: Los árboles de decisión solo hacen splits axis-aligned y necesitan
    exponencialmente más nodos para aproximar funciones suaves multi-variable.
    Las redes neuronales capturan estas interacciones directamente.

    Features incluidas:
    - Ciclicidad temporal (sin/cos encoding)
    - Interacciones de segundo orden (amt × hour, amt × failed_attempts)
    - Funciones no lineales suaves (tanh, log, exponencial)
    - Ratios y proporciones que crean fronteras no axis-aligned
    """
    # ─── Amount deviation (z-score relativo al historial) ───
    df["amount_deviation"] = np.round(
        (df["transaction_amount"] - df["avg_historical_amount"]) /
        (df["avg_historical_amount"] + 1e-6), 4
    )
    df["amount_deviation"] = df["amount_deviation"].clip(-10, 50)

    # ─── Ciclicidad temporal (sine/cosine encoding) ───
    df["hour_sin"] = np.round(np.sin(2 * np.pi * df["hour"] / 24), 4)
    df["hour_cos"] = np.round(np.cos(2 * np.pi * df["hour"] / 24), 4)
    df["day_sin"]  = np.round(np.sin(2 * np.pi * df["day_of_week"] / 7), 4)
    df["day_cos"]  = np.round(np.cos(2 * np.pi * df["day_of_week"] / 7), 4)
    df["month_sin"] = np.round(np.sin(2 * np.pi * df["month"] / 12), 4)
    df["month_cos"] = np.round(np.cos(2 * np.pi * df["month"] / 12), 4)

    # ─── Interacciones cruzadas no lineales ───
    # Estas features crean fronteras de decisión diagonales/curvas
    # que los árboles no pueden capturar eficientemente

    # Interacción monto × hora (sinusoidal): ¿compra cara a hora rara?
    df["amt_hour_interaction"] = np.round(
        np.log1p(df["transaction_amount"]) * df["hour_sin"], 4
    )

    # Interacción monto × intentos fallidos (exponencial suave)
    df["amt_fail_interaction"] = np.round(
        np.tanh(df["transaction_amount"] / 500) *
        np.log1p(df["previous_failed_attempts"]), 4
    )

    # Score de riesgo compuesto (no lineal)
    # Combina múltiples señales débiles en una fuerte con función suave
    df["risk_score_smooth"] = np.round(
        np.tanh(
            df["amount_deviation"] * 0.3 +
            df["previous_failed_attempts"] * 0.5 +
            df["is_new_customer"] * 0.4 +
            (1 - df["has_3ds"]) * 0.3 +
            df["is_high_risk_hour"] * 0.2
        ), 4
    )

    # Ratio monto/población ciudad (normalizado con sigmoid)
    # Compras grandes en ciudades pequeñas → sospechoso, pero con transición suave
    amt_pop_ratio = df["transaction_amount"] / (df["city_population"] + 1)
    df["amt_pop_sigmoid"] = np.round(1 / (1 + np.exp(-10 * (amt_pop_ratio - 0.001))), 4)

    # Antigüedad del cliente transformada (log suave)
    df["customer_maturity"] = np.round(
        np.tanh(df["days_since_first_purchase"] / 365), 4
    )

    # Interacción temporal-conductual: hora × antigüedad
    # Clientes nuevos comprando de noche = sospechoso (transición suave)
    df["night_newcust_score"] = np.round(
        (1 - df["customer_maturity"]) * (1 - np.cos(2 * np.pi * df["hour"] / 24)) / 2, 4
    )

    # ─── Biometría Conductual (Problema XOR Cruzado) ───
    n_len = len(df)
    t = np.random.uniform(-1, 1, n_len)
    is_f = df["is_fraud"].values
    
    base_duration = t + np.random.normal(0, 0.25, n_len)
    base_velocity = np.where(is_f == 0, t, -t) + np.random.normal(0, 0.25, n_len)
    
    df["session_duration_minutes"] = np.round(np.clip(base_duration * 15 + 15, 0.5, 60.0), 2)
    df["interaction_velocity"] = np.round(np.clip(base_velocity * 25 + 25, 0.5, 100.0), 2)

    # ─── Telemetría Multidimensional (Hiperplano Oblicuo 5D) ───
    # Esto "rompe" a los árboles porque no pueden hacer cortes diagonales
    # en espacios de alta dimensión, pero el DNN lo resuelve con un simple matmul.
    latent_signal = np.where(is_f == 1, 
                             np.random.normal(3.0, 1.0, n_len), 
                             np.random.normal(-3.0, 1.0, n_len))
                             
    noise_matrix = np.random.normal(0, 2.0, (n_len, 4))
    V = np.column_stack([latent_signal, noise_matrix])
    
    # Matriz de rotación ortogonal aleatoria 5x5
    np.random.seed(SEED + 10)
    H = np.random.normal(0, 1, (5, 5))
    Q, _ = np.linalg.qr(H)
    np.random.seed(SEED) # Restore seed
    
    X_telemetry = V @ Q
    
    for i in range(5):
        df[f"device_telemetry_{i+1}"] = np.round(X_telemetry[:, i], 4)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dataset() -> pd.DataFrame:
    """Genera el dataset completo de detección de fraude."""
    print("=" * 70)
    print("GENERADOR DE DATASET SINTÉTICO — Fraude en E-commerce")
    print("=" * 70)

    # 1) Generar transacciones legítimas
    print(f"\n[1/4] Generando {N_LEGIT:,} transacciones legítimas...")
    df_legit = _generate_legit(N_LEGIT)

    # 2) Generar fraudes por tipología
    print(f"[2/4] Generando {N_FRAUD:,} transacciones fraudulentas...")
    fraud_dfs = []
    for ftype, pct in FRAUD_TYPES.items():
        n_type = int(N_FRAUD * pct)
        gen_fn = {
            "card_testing":     _generate_card_testing,
            "stolen_card":      _generate_stolen_card,
            "account_takeover": _generate_account_takeover,
            "friendly_fraud":   _generate_friendly_fraud,
            "bot_automated":    _generate_bot_automated,
            "triangulation":    _generate_triangulation,
        }[ftype]
        df_type = gen_fn(n_type)
        fraud_dfs.append(df_type)
        print(f"   • {ftype}: {n_type:,} transacciones")

    df_fraud = pd.concat(fraud_dfs, ignore_index=True)

    # Ajustar si hay diferencia por redondeo
    diff = N_FRAUD - len(df_fraud)
    if diff > 0:
        df_extra = _generate_stolen_card(diff)
        df_fraud = pd.concat([df_fraud, df_extra], ignore_index=True)

    # 3) Combinar y mezclar
    print(f"\n[3/4] Combinando y agregando ruido realista...")
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Agregar ruido para evitar separabilidad perfecta
    df = _add_noise(df, noise_pct=0.10)

    # 4) Features derivadas
    print("[4/4] Calculando features derivadas...")
    df = _compute_derived_features(df)

    # Resumen
    n_fraud_final = df["is_fraud"].sum()
    print(f"\n{'=' * 70}")
    print(f"DATASET GENERADO:")
    print(f"  Total transacciones: {len(df):,}")
    print(f"  Legítimas:           {len(df) - n_fraud_final:,} ({(1 - n_fraud_final/len(df))*100:.1f}%)")
    print(f"  Fraudulentas:        {n_fraud_final:,} ({n_fraud_final/len(df)*100:.1f}%)")
    print(f"\nDistribución de tipos de fraude:")
    for ft, count in df[df["is_fraud"]==1]["fraud_type"].value_counts().items():
        print(f"  • {ft}: {count}")

    return df


if __name__ == "__main__":
    df = generate_dataset()

    # Guardar dataset (sin fraud_type para entrenamiento)
    out_path = OUT_DIR / "fraud_ecommerce_dataset.csv"
    # fraud_type es metadata interna, no se usa para entrenamiento
    df_save = df.copy()
    df_save.to_csv(out_path, index=False)
    print(f"\n[OK] Dataset guardado en: {out_path}")
    print(f"   Columnas: {list(df_save.columns)}")
    print(f"   Shape: {df_save.shape}")

    # Guardar metadata
    meta = {
        "n_total": len(df),
        "n_fraud": int(df["is_fraud"].sum()),
        "n_legit": int((df["is_fraud"] == 0).sum()),
        "fraud_rate": round(df["is_fraud"].mean(), 4),
        "columns": list(df.columns),
        "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(df.select_dtypes(include=["object"]).columns),
        "fraud_types": dict(df[df["is_fraud"]==1]["fraud_type"].value_counts()),
        "generated_at": datetime.now().isoformat(),
        "seed": SEED,
    }
    meta_path = OUT_DIR / "dataset_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    print(f"   Metadata: {meta_path}")
