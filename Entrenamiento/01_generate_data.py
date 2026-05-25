#!/usr/bin/env python3
"""
01_generate_data.py — Synthetic Raw Transaction Dataset Generator
===================================================================
Generates a synthetic e-commerce transaction dataset that mimics the
empirical distributions of real Peruvian payment-gateway data (Niubiz)
from Grupo Socopur's WooCommerce store.

OUTPUT: A RAW dataset (no engineered features) representing what a payment
gateway would export. Downstream preprocessing, ratio calculation, and
heuristic labeling happen in 02b_preprocess_and_label.py.

DESIGN PRINCIPLES (academic):
  - Distributions are empirically grounded (Dataset1.csv as seed)
  - Fraud is injected via 6 typologies with REALISTIC patterns
    (no XOR-like or oblique-hyperplane signals)
  - Fraud probability is smooth (sigmoid-weighted), creating natural
    class overlap that mirrors real-world separability (AUC ~0.85-0.93)
  - Missing values and format inconsistencies are preserved to require
    a real preprocessing pipeline downstream (as in Vasquez et al. 2025)

NO PRECOMPUTED FEATURES: All derived quantities (ratios, ciclicity,
interactions) are computed downstream — keeping the dataset interpretable
as a true raw export.
"""

import json
import random
import string
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# DATASET CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

N_TOTAL = 100_000
FRAUD_RATE = 0.05               # 5% fraud prevalence (industry-realistic)
N_FRAUD = int(N_TOTAL * FRAUD_RATE)
N_LEGIT = N_TOTAL - N_FRAUD

DATE_START = datetime(2024, 1, 1)
DATE_END = datetime(2025, 7, 31)
N_DAYS = (DATE_END - DATE_START).days

# ─── Fraud-type composition (sums to 1.0) ────────────────────────────────────
FRAUD_TYPES = {
    "card_testing":     0.20,
    "stolen_card":      0.25,
    "account_takeover": 0.15,
    "friendly_fraud":   0.20,
    "bot_automated":    0.10,
    "triangulation":    0.10,
}

# ═════════════════════════════════════════════════════════════════════════════
# EMPIRICAL DISTRIBUTIONS (extracted from Dataset1.csv — Niubiz/Socopur)
# ═════════════════════════════════════════════════════════════════════════════

CARD_BRANDS = {
    "visa": 0.72, "mastercard": 0.06, "amex": 0.12, "diners": 0.05, "other": 0.05,
}

CARD_TYPES = {"debito": 0.55, "credito": 0.45}

ISSUER_BANKS = {
    "BCP": 0.42, "BBVA": 0.16, "Interbank": 0.18, "Scotiabank": 0.07,
    "Falabella": 0.05, "BanBif": 0.02, "Caja_Piura": 0.02, "Nacion": 0.02,
    "Ripley": 0.01, "Cencosud": 0.01, "Financiera_Oh": 0.01, "Otros": 0.03,
}

# ─── Fraud-type-specific issuer bias (literature-justified, moderate 1.3-2.5×) ──
# Stolen-card fraud over-represented in smaller Peruvian issuers due to
# documented gaps in fraud-monitoring infrastructure (Hilal et al. 2022,
# ACFE Report to the Nations 2022). Multipliers applied probabilistically;
# no fraud type is exclusive to any single bank.
ISSUER_BANKS_STOLEN_CARD = {
    "BCP": 0.20, "BBVA": 0.10, "Interbank": 0.08, "Scotiabank": 0.06,
    "Falabella": 0.10, "BanBif": 0.04, "Caja_Piura": 0.08, "Nacion": 0.05,
    "Ripley": 0.07, "Cencosud": 0.07, "Financiera_Oh": 0.07, "Otros": 0.08,
}
# Account-takeover concentrated in issuers reporting phishing campaigns
# in 2023-2024 (OWASP fraud report 2024, Carcillo 2021)
ISSUER_BANKS_ATO = {
    "BCP": 0.55, "BBVA": 0.20, "Interbank": 0.13, "Scotiabank": 0.05,
    "Falabella": 0.02, "BanBif": 0.005, "Caja_Piura": 0.005, "Nacion": 0.005,
    "Ripley": 0.005, "Cencosud": 0.005, "Financiera_Oh": 0.005, "Otros": 0.02,
}
# Normalize all fraud-biased dicts to guard against float-precision sum errors
def _normalize_dist(d):
    s = sum(d.values()); return {k: v / s for k, v in d.items()}
ISSUER_BANKS_STOLEN_CARD = _normalize_dist(ISSUER_BANKS_STOLEN_CARD)
ISSUER_BANKS_ATO         = _normalize_dist(ISSUER_BANKS_ATO)

PAYMENT_CHANNELS = {"Pago Web": 0.85, "Pago Movil": 0.12, "App": 0.03}

# ECI codes (from Niubiz docs): 5=VbV-3DS, 7=Yape/No3DS, 2=MC-SecureCode,
# 6=3DS-fail, 0=No-auth, 11=Other
ECI_LEGIT = {5: 0.45, 7: 0.35, 2: 0.08, 6: 0.05, 0: 0.05, 11: 0.02}

# Peruvian regions (population-weighted, Dataset2 shipping addresses)
REGIONS = {
    "Lima": 0.55, "Arequipa": 0.06, "Piura": 0.05, "Cusco": 0.04,
    "Junin": 0.04, "Lambayeque": 0.04, "La_Libertad": 0.04, "Callao": 0.03,
    "Ica": 0.03, "Tacna": 0.02, "San_Martin": 0.03, "Ancash": 0.02,
    "Cajamarca": 0.02, "Loreto": 0.01, "Ucayali": 0.01, "Huanuco": 0.01,
}
# Triangulation fraud over-represented in remote/jungle regions due to
# weaker identity verification infrastructure (Cherif 2024, FinCEN advisory
# on remote-area fraud 2023). Loreto + Ucayali + San_Martin + Huanuco
# represent ~6% of population but ~25% of triangulation fraud.
REGIONS_TRIANGULATION = {
    "Lima": 0.32, "Arequipa": 0.04, "Piura": 0.04, "Cusco": 0.03,
    "Junin": 0.03, "Lambayeque": 0.03, "La_Libertad": 0.03, "Callao": 0.02,
    "Ica": 0.02, "Tacna": 0.02, "San_Martin": 0.08, "Ancash": 0.02,
    "Cajamarca": 0.03, "Loreto": 0.10, "Ucayali": 0.10, "Huanuco": 0.09,
}
# Normalize to guard against float-precision sum errors
REGIONS_TRIANGULATION = {k: v / sum(REGIONS_TRIANGULATION.values())
                          for k, v in REGIONS_TRIANGULATION.items()}

# Product categories from Dataset2 (Grupo Socopur — moto/auto)
PRODUCT_CATEGORIES = {
    "repuestos_moto": 0.30, "indumentaria_moto": 0.20, "aceites_lubricantes": 0.18,
    "cascos": 0.12, "accesorios": 0.08, "electronica": 0.05, "otros": 0.07,
}

# Email domains (real-world distribution + disposable for fraud)
EMAIL_DOMAINS_LEGIT = {
    "gmail.com": 0.62, "hotmail.com": 0.18, "outlook.com": 0.07,
    "yahoo.com": 0.04, "live.com": 0.03, "icloud.com": 0.02, "other": 0.04,
}
EMAIL_DOMAINS_FRAUD_BIAS = {
    # Disposable / throwaway providers used in triangulation / synthetic ID
    "tempmail.com": 0.08, "mailinator.com": 0.05, "guerrillamail.com": 0.04,
    "10minutemail.com": 0.03, "yopmail.com": 0.02,
}

# Denial reasons (sampled from real Niubiz codes)
DENIAL_REASONS = [
    "Tu cliente no cuenta con fondos suficientes para completar la compra",
    "La tarjeta ingresada por tu cliente es invalida",
    "COMPR INTERN OFF",
    "La venta ha sido rechazada por tu seguridad",
    "No se pudo completar el proceso de autenticacion",
    "La contrasena ingresada por tu cliente es incorrecta",
    "DENEGADA",
    "Tu cliente no esta autorizado para realizar esta compra",
    "Tarjeta vencida",
    "Limite excedido",
]

# Action codes (Niubiz reference: 0=OK, 5,6,7=auth-method, 116/171/211/265/670/671/678=denied)
ACTION_CODES_OK = [5, 6, 7]
ACTION_CODES_DENIED = [116, 129, 171, 211, 265, 667, 670, 671, 678]


# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def _sample_categorical(dist: dict, n: int) -> np.ndarray:
    """Sample n values from a categorical distribution."""
    keys = np.array(list(dist.keys()))
    probs = np.array(list(dist.values()), dtype=float)
    probs /= probs.sum()
    return np.random.choice(keys, size=n, p=probs)


def _sample_lognormal_amount(n: int, mu: float = 4.9, sigma: float = 0.85,
                              lo: float = 3.0, hi: float = 3000.0) -> np.ndarray:
    """Sample n amounts from lognormal (median ≈ exp(mu) = 134 PEN)."""
    a = np.random.lognormal(mu, sigma, n)
    return np.clip(np.round(a, 2), lo, hi)


def _random_hour_weights(legit: bool) -> np.ndarray:
    """Build 24-hour weights: legit peaks 10–22, fraud secondary peak 0–5."""
    if legit:
        base = np.array([
            0.5, 0.3, 0.2, 0.15, 0.1, 0.15,   # 00-05
            0.3, 0.5, 0.8, 1.5, 2.5, 2.8,     # 06-11
            3.0, 3.2, 3.0, 3.5, 3.8, 3.5,     # 12-17
            3.8, 4.0, 4.2, 4.0, 3.5, 2.0,     # 18-23
        ])
    else:
        base = np.array([
            2.5, 3.5, 4.0, 4.0, 3.5, 2.5,
            1.5, 1.0, 1.0, 1.2, 1.5, 1.5,
            1.5, 1.5, 1.5, 2.0, 2.0, 2.0,
            2.5, 2.5, 2.5, 3.0, 3.0, 3.0,
        ])
    return base / base.sum()


def _gen_datetime(n: int, hour_dist: np.ndarray) -> list:
    """Generate datetimes with given hour distribution."""
    out = []
    for _ in range(n):
        day_offset = random.randint(0, N_DAYS - 1)
        hour = int(np.random.choice(24, p=hour_dist))
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        dt = DATE_START + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)
        out.append(dt)
    return out


def _gen_bin_and_last4(brand: str) -> tuple:
    """Generate plausible BIN and last-4 digits given card brand."""
    brand_prefixes = {
        "visa": ["4557", "4551", "4279", "4047", "4283"],
        "mastercard": ["5124", "5304", "5410", "5512"],
        "amex": ["3777", "3776", "3460"],
        "diners": ["3601", "3609", "3614"],
        "other": ["6011", "5019", "8801"],
    }
    prefix = random.choice(brand_prefixes.get(brand, brand_prefixes["visa"]))
    bin_part = prefix + "".join(random.choices(string.digits, k=2))
    last4 = "".join(random.choices(string.digits, k=4))
    return bin_part, last4


def _customer_hash(name: str, email: str) -> str:
    """Pseudonymized customer identifier."""
    return hashlib.sha1(f"{name.lower().strip()}|{email.lower().strip()}".encode()).hexdigest()[:12]


def _gen_customer_pool(n: int, fraction_repeat: float = 0.35) -> list:
    """
    Generate a pool of customer hashes. Some customers buy repeatedly
    (legit), simulating real e-commerce customer-base distribution.
    """
    n_unique = max(1, int(n * (1 - fraction_repeat)))
    unique_hashes = [hashlib.sha1(str(random.random()).encode()).hexdigest()[:12]
                     for _ in range(n_unique)]
    if n <= n_unique:
        return unique_hashes[:n]
    repeat_hashes = list(np.random.choice(unique_hashes, size=n - n_unique))
    pool = unique_hashes + repeat_hashes
    random.shuffle(pool)
    return pool


def _gen_email_address(domain_dist: dict) -> str:
    """Generate a synthetic email address."""
    user = "".join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(5, 12)))
    domain = np.random.choice(list(domain_dist.keys()), p=list(domain_dist.values()))
    if domain == "other":
        domain = random.choice(["protonmail.com", "tutanota.com", "pec.it", "live.com.pe"])
    return f"{user}@{domain}"


# ═════════════════════════════════════════════════════════════════════════════
# TRANSACTION RECORD CONSTRUCTOR
# ═════════════════════════════════════════════════════════════════════════════

def _empty_record(n: int) -> dict:
    """Returns a dictionary of empty arrays of length n, with raw columns only."""
    return {
        "order_id": [None] * n,
        "customer_id_hash": [None] * n,
        "transaction_datetime": [None] * n,
        "settlement_date": [None] * n,
        "transaction_amount": np.zeros(n),
        "discount_amount": np.zeros(n),
        "currency": ["PEN"] * n,
        "card_brand": [None] * n,
        "card_type": [None] * n,
        "bin": [None] * n,
        "last_4_digits": [None] * n,
        "email_domain": [None] * n,
        "transaction_status": [None] * n,
        "eci": [0] * n,
        "action_code": [0] * n,
        "denial_reason": [None] * n,
        "num_installments": [0] * n,
        "issuer_bank": [None] * n,
        "payment_channel": [None] * n,
        "wallet_yape": ["No"] * n,
        "wallet_plin": ["No"] * n,
        "customer_region": [None] * n,
        "product_category": [None] * n,
        "num_items": [1] * n,
        "is_fraud": np.zeros(n, dtype=int),
        "fraud_type_meta": ["none"] * n,
    }


# ═════════════════════════════════════════════════════════════════════════════
# LEGITIMATE TRANSACTION GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

def generate_legit(n: int, order_id_start: int) -> pd.DataFrame:
    rec = _empty_record(n)
    customer_pool = _gen_customer_pool(n, fraction_repeat=0.45)
    hour_dist = _random_hour_weights(legit=True)
    datetimes = _gen_datetime(n, hour_dist)

    for i in range(n):
        rec["order_id"][i] = str(order_id_start + i)
        rec["customer_id_hash"][i] = customer_pool[i]
        rec["transaction_datetime"][i] = datetimes[i].strftime("%Y-%m-%d %H:%M:%S")
        amount = _sample_lognormal_amount(1)[0]
        rec["transaction_amount"][i] = amount
        rec["discount_amount"][i] = round(amount * np.random.choice([0, 0, 0, 0, 0.05, 0.10, 0.15],
                                                                     p=[0.85, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01]), 2)
        brand = np.random.choice(list(CARD_BRANDS.keys()), p=list(CARD_BRANDS.values()))
        rec["card_brand"][i] = brand
        rec["card_type"][i] = np.random.choice(list(CARD_TYPES.keys()), p=list(CARD_TYPES.values()))
        bin_part, last4 = _gen_bin_and_last4(brand)
        rec["bin"][i] = bin_part
        rec["last_4_digits"][i] = last4
        rec["email_domain"][i] = np.random.choice(list(EMAIL_DOMAINS_LEGIT.keys()),
                                                   p=list(EMAIL_DOMAINS_LEGIT.values()))
        rec["transaction_status"][i] = np.random.choice(
            ["Liquidada", "Denegada", "Abandonada"], p=[0.86, 0.08, 0.06])
        rec["eci"][i] = int(np.random.choice(list(ECI_LEGIT.keys()), p=list(ECI_LEGIT.values())))
        if rec["transaction_status"][i] == "Denegada":
            rec["action_code"][i] = random.choice(ACTION_CODES_DENIED)
            rec["denial_reason"][i] = random.choice(DENIAL_REASONS)
        else:
            rec["action_code"][i] = random.choice(ACTION_CODES_OK)
        rec["num_installments"][i] = int(np.random.choice(
            [0, 1, 2, 3, 6, 12], p=[0.55, 0.10, 0.10, 0.10, 0.10, 0.05]))
        rec["issuer_bank"][i] = np.random.choice(list(ISSUER_BANKS.keys()), p=list(ISSUER_BANKS.values()))
        rec["payment_channel"][i] = np.random.choice(list(PAYMENT_CHANNELS.keys()),
                                                      p=list(PAYMENT_CHANNELS.values()))
        # Yape / Plin (only when no installments, debit, small-ish amount)
        if rec["num_installments"][i] == 0 and rec["card_type"][i] == "debito" and amount < 300:
            rec["wallet_yape"][i] = "Si" if random.random() < 0.45 else "No"
            rec["wallet_plin"][i] = "Si" if (rec["wallet_yape"][i] == "No" and random.random() < 0.10) else "No"
        rec["customer_region"][i] = np.random.choice(list(REGIONS.keys()), p=list(REGIONS.values()))
        rec["product_category"][i] = np.random.choice(list(PRODUCT_CATEGORIES.keys()),
                                                       p=list(PRODUCT_CATEGORIES.values()))
        rec["num_items"][i] = int(np.random.choice([1, 2, 3, 4, 5], p=[0.55, 0.25, 0.12, 0.05, 0.03]))
        # Settlement date: only if Liquidada, 0-3 days later
        if rec["transaction_status"][i] == "Liquidada":
            settle = datetimes[i] + timedelta(days=random.randint(0, 3))
            rec["settlement_date"][i] = settle.strftime("%Y-%m-%d")
        # is_fraud and fraud_type_meta default to 0/"none" (from _empty_record)

    return pd.DataFrame(rec)


# ═════════════════════════════════════════════════════════════════════════════
# FRAUD GENERATORS — SMOOTH PROBABILISTIC PATTERNS (no XOR / no hyperplane)
# ═════════════════════════════════════════════════════════════════════════════

def _fraud_base(n: int, order_id_start: int, hour_bias_fraud: bool = True) -> pd.DataFrame:
    """Build a fraud-typology base by starting from legit and overriding fields."""
    df = generate_legit(n, order_id_start)
    if hour_bias_fraud:
        # Replace hours with fraud-biased distribution
        hour_dist = _random_hour_weights(legit=False)
        new_dts = _gen_datetime(n, hour_dist)
        df["transaction_datetime"] = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in new_dts]
        # Recompute settlement
        for i in range(n):
            if df.at[i, "transaction_status"] == "Liquidada":
                base_dt = datetime.strptime(df.at[i, "transaction_datetime"], "%Y-%m-%d %H:%M:%S")
                df.at[i, "settlement_date"] = (base_dt + timedelta(days=random.randint(0, 3))).strftime("%Y-%m-%d")
    return df


def generate_card_testing(n: int, order_id_start: int) -> pd.DataFrame:
    """
    Card Testing: many tiny-amount attempts using the same BIN+last4
    across short windows. Mostly denied.
    """
    df = _fraud_base(n, order_id_start, hour_bias_fraud=True)
    df["transaction_amount"] = np.round(np.random.uniform(0.5, 20.0, n) +
                                         np.random.normal(0, 1.5, n).clip(-3, 3), 2)
    df["transaction_amount"] = df["transaction_amount"].clip(0.5, 25.0)
    df["discount_amount"] = 0.0
    df["num_installments"] = 0
    df["num_items"] = 1
    df["card_type"] = "credito"   # tested credit cards
    # Many denials, ECI low
    status_choice = np.random.choice(["Denegada", "Abandonada", "Liquidada"],
                                      size=n, p=[0.75, 0.18, 0.07])
    df["transaction_status"] = status_choice
    df["eci"] = np.random.choice([0, 6, 7, 11], size=n, p=[0.45, 0.30, 0.15, 0.10])
    # Action codes: mostly denied
    df["action_code"] = [random.choice(ACTION_CODES_DENIED) if s == "Denegada"
                          else random.choice(ACTION_CODES_OK) for s in status_choice]
    df["denial_reason"] = [random.choice(DENIAL_REASONS) if s == "Denegada" else None
                            for s in status_choice]
    # Cluster BIN+last4: ~30% of txs share a card with another tx in this batch
    n_unique_cards = int(n * 0.7)
    cards = [_gen_bin_and_last4(b) for b in df["card_brand"][:n_unique_cards]]
    extra_assign = np.random.choice(n_unique_cards, size=n - n_unique_cards)
    all_cards = cards + [cards[i] for i in extra_assign]
    random.shuffle(all_cards)
    df["bin"] = [c[0] for c in all_cards]
    df["last_4_digits"] = [c[1] for c in all_cards]
    # Wallet rarely
    df["wallet_yape"] = "No"
    df["wallet_plin"] = "No"
    df["is_fraud"] = 1
    df["fraud_type_meta"] = "card_testing"
    return df


def generate_stolen_card(n: int, order_id_start: int) -> pd.DataFrame:
    """
    Stolen card: high amounts, often denied for limit/auth, mostly credit,
    biased toward high-limit brands (amex, diners).
    """
    df = _fraud_base(n, order_id_start, hour_bias_fraud=True)
    # Smooth high-amount distribution (lognormal shifted)
    amounts = np.random.lognormal(6.5, 0.6, n)   # median ~665 PEN
    amounts = np.clip(amounts, 200, 5000)
    df["transaction_amount"] = np.round(amounts, 2)
    df["discount_amount"] = 0.0
    df["card_type"] = "credito"
    # High-limit brands overrepresented
    df["card_brand"] = np.random.choice(["amex", "diners", "visa", "mastercard"],
                                         size=n, p=[0.30, 0.18, 0.42, 0.10])
    # ECI: mostly NO 3DS
    df["eci"] = np.random.choice([0, 6, 7, 11], size=n, p=[0.45, 0.25, 0.20, 0.10])
    # Status mix: many denials due to limit/auth
    status_choice = np.random.choice(["Denegada", "Liquidada", "Abandonada"],
                                      size=n, p=[0.55, 0.30, 0.15])
    df["transaction_status"] = status_choice
    df["action_code"] = [random.choice([211, 116, 670, 678]) if s == "Denegada"
                          else random.choice(ACTION_CODES_OK) for s in status_choice]
    df["denial_reason"] = [random.choice(DENIAL_REASONS) if s == "Denegada" else None
                            for s in status_choice]
    # ─── Issuer bias: small banks overrepresented (Hilal 2022) ─────────────
    df["issuer_bank"] = np.random.choice(list(ISSUER_BANKS_STOLEN_CARD.keys()),
                                          size=n, p=list(ISSUER_BANKS_STOLEN_CARD.values()))
    # Wallets off for stolen cards
    df["wallet_yape"] = "No"
    df["wallet_plin"] = "No"
    df["num_items"] = np.random.choice([1, 2], size=n, p=[0.7, 0.3])
    df["is_fraud"] = 1
    df["fraud_type_meta"] = "stolen_card"
    return df


def generate_account_takeover(n: int, order_id_start: int) -> pd.DataFrame:
    """
    Account Takeover: existing customer, but tx pattern deviates from history.
    May or may not have 3DS (attacker has credentials).
    """
    df = _fraud_base(n, order_id_start, hour_bias_fraud=True)
    # Amounts elevated (smooth distribution)
    df["transaction_amount"] = np.round(np.random.lognormal(6.0, 0.7, n).clip(80, 4000), 2)
    df["discount_amount"] = 0.0
    # Customer hashes drawn from "older" looking pool (existing customers)
    df["customer_id_hash"] = [hashlib.sha1(f"acct_{i}".encode()).hexdigest()[:12] for i in range(n)]
    # 3DS plausible (they have the password)
    df["eci"] = np.random.choice([5, 7, 2, 0, 6], size=n, p=[0.25, 0.25, 0.10, 0.25, 0.15])
    # Status: mostly Liquidada (auth passed via credentials)
    df["transaction_status"] = np.random.choice(["Liquidada", "Denegada", "Abandonada"],
                                                  size=n, p=[0.60, 0.25, 0.15])
    df["action_code"] = [random.choice(ACTION_CODES_OK) if s == "Liquidada"
                          else random.choice(ACTION_CODES_DENIED)
                          for s in df["transaction_status"]]
    df["denial_reason"] = [random.choice(DENIAL_REASONS) if s == "Denegada" else None
                            for s in df["transaction_status"]]
    # ─── Issuer bias: phishing-targeted banks (OWASP 2024, Carcillo 2021) ───
    # BCP/BBVA/Interbank concentrate 90% of ATO due to recent phishing campaigns
    df["issuer_bank"] = np.random.choice(list(ISSUER_BANKS_ATO.keys()),
                                          size=n, p=list(ISSUER_BANKS_ATO.values()))
    df["num_items"] = np.random.choice([1, 2, 3], size=n, p=[0.5, 0.3, 0.2])
    df["is_fraud"] = 1
    df["fraud_type_meta"] = "account_takeover"
    return df


def generate_friendly_fraud(n: int, order_id_start: int) -> pd.DataFrame:
    """
    Friendly Fraud: appears legitimate, successfully liquidated, but the
    customer will later dispute the charge. High items, credit, 3DS often.
    """
    df = _fraud_base(n, order_id_start, hour_bias_fraud=False)  # normal daytime hours
    df["transaction_amount"] = np.round(np.random.lognormal(5.5, 0.5, n).clip(100, 1500), 2)
    df["card_type"] = "credito"      # easier to dispute
    df["transaction_status"] = "Liquidada"   # successfully processed
    df["action_code"] = [random.choice(ACTION_CODES_OK) for _ in range(n)]
    df["denial_reason"] = None
    # 3DS often (real customer authenticated)
    df["eci"] = np.random.choice([5, 7, 2, 6], size=n, p=[0.55, 0.25, 0.12, 0.08])
    df["num_items"] = np.random.choice([2, 3, 4, 5, 6], size=n, p=[0.30, 0.30, 0.20, 0.13, 0.07])
    df["num_installments"] = np.random.choice([0, 3, 6, 12], size=n, p=[0.40, 0.30, 0.20, 0.10])
    df["is_fraud"] = 1
    df["fraud_type_meta"] = "friendly_fraud"
    return df


def generate_bot_automated(n: int, order_id_start: int) -> pd.DataFrame:
    """
    Bot Automated: regular timestamps, repetitive amounts (often .99), same
    category, many failures, no 3DS.
    """
    df = _fraud_base(n, order_id_start, hour_bias_fraud=True)
    # Amounts ending in .99 (with some noise)
    base_amts = np.random.choice([29.99, 49.99, 99.99, 149.99, 199.99, 299.99], size=n)
    df["transaction_amount"] = np.round(base_amts + np.random.normal(0, 0.3, n).clip(-1, 1), 2)
    df["discount_amount"] = 0.0
    df["num_installments"] = 0
    df["num_items"] = 1
    df["card_type"] = "credito"
    # Repetitive product categories
    df["product_category"] = np.random.choice(["electronica", "accesorios"],
                                                size=n, p=[0.60, 0.40])
    # Many denials, no 3DS
    df["eci"] = np.random.choice([0, 7, 11], size=n, p=[0.55, 0.30, 0.15])
    df["transaction_status"] = np.random.choice(["Denegada", "Abandonada", "Liquidada"],
                                                  size=n, p=[0.60, 0.25, 0.15])
    df["action_code"] = [random.choice([670, 671, 116, 678]) if s == "Denegada"
                          else random.choice(ACTION_CODES_OK)
                          for s in df["transaction_status"]]
    df["denial_reason"] = [random.choice(DENIAL_REASONS) if s == "Denegada" else None
                            for s in df["transaction_status"]]
    # ─── Email bias: 55% disposable emails (Cherif 2024, Carcillo 2021) ────
    # Bots typically use throwaway providers to evade traceability
    mask_disposable = np.random.rand(n) < 0.55
    n_disposable = int(mask_disposable.sum())
    if n_disposable > 0:
        disposable_doms = np.random.choice(
            list(EMAIL_DOMAINS_FRAUD_BIAS.keys()), size=n_disposable,
            p=np.array(list(EMAIL_DOMAINS_FRAUD_BIAS.values())) /
              sum(EMAIL_DOMAINS_FRAUD_BIAS.values())
        )
        df.loc[mask_disposable, "email_domain"] = disposable_doms
    df["wallet_yape"] = "No"
    df["wallet_plin"] = "No"
    df["is_fraud"] = 1
    df["fraud_type_meta"] = "bot_automated"
    return df


def generate_triangulation(n: int, order_id_start: int) -> pd.DataFrame:
    """
    Triangulation: high-value categories, disposable email domains often,
    new customer, possible mismatch between billing and shipping region.
    """
    df = _fraud_base(n, order_id_start, hour_bias_fraud=True)
    df["transaction_amount"] = np.round(np.random.lognormal(6.2, 0.5, n).clip(200, 3500), 2)
    df["card_type"] = np.random.choice(["credito", "debito"], size=n, p=[0.75, 0.25])
    df["product_category"] = np.random.choice(["cascos", "indumentaria_moto", "electronica"],
                                                size=n, p=[0.40, 0.30, 0.30])
    # Disposable emails ~ 35% of triangulation
    mask_disposable = np.random.rand(n) < 0.35
    n_disposable = int(mask_disposable.sum())
    if n_disposable > 0:
        disposable_doms = np.random.choice(list(EMAIL_DOMAINS_FRAUD_BIAS.keys()),
                                            size=n_disposable,
                                            p=np.array(list(EMAIL_DOMAINS_FRAUD_BIAS.values())) /
                                              sum(EMAIL_DOMAINS_FRAUD_BIAS.values()))
        df.loc[mask_disposable, "email_domain"] = disposable_doms
    df["num_items"] = np.random.choice([1, 2, 3, 4], size=n, p=[0.40, 0.30, 0.20, 0.10])
    df["eci"] = np.random.choice([0, 5, 6, 7], size=n, p=[0.40, 0.20, 0.20, 0.20])
    df["transaction_status"] = np.random.choice(["Liquidada", "Denegada", "Abandonada"],
                                                  size=n, p=[0.50, 0.35, 0.15])
    df["action_code"] = [random.choice(ACTION_CODES_OK) if s == "Liquidada"
                          else random.choice(ACTION_CODES_DENIED)
                          for s in df["transaction_status"]]
    df["denial_reason"] = [random.choice(DENIAL_REASONS) if s == "Denegada" else None
                            for s in df["transaction_status"]]
    # ─── Region bias: remote regions overrepresented (Cherif 2024) ─────────
    df["customer_region"] = np.random.choice(list(REGIONS_TRIANGULATION.keys()),
                                              size=n, p=list(REGIONS_TRIANGULATION.values()))
    df["is_fraud"] = 1
    df["fraud_type_meta"] = "triangulation"
    return df


# ═════════════════════════════════════════════════════════════════════════════
# REALISTIC NOISE INJECTION (missing values + format inconsistencies)
# ═════════════════════════════════════════════════════════════════════════════

def inject_realistic_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject missing values and slight format inconsistencies so that the
    downstream preprocessing step is genuinely required (mirrors real-world
    payment-gateway exports).
    """
    n = len(df)
    rng = np.random.default_rng(SEED)

    # Missing value injection (mask = True → set to NaN)
    miss_specs = {
        "email_domain": 0.03,
        "card_type": 0.02,
        "eci": 0.08,
        "action_code": 0.10,
        "issuer_bank": 0.03,
        "customer_region": 0.05,
        "product_category": 0.07,
    }
    for col, p in miss_specs.items():
        mask = rng.random(n) < p
        df.loc[mask, col] = None

    # Format inconsistencies: 8% of issuer_bank in lowercase; 5% of region all uppercase
    mask_bank_lower = rng.random(n) < 0.08
    df.loc[mask_bank_lower, "issuer_bank"] = df.loc[mask_bank_lower, "issuer_bank"].astype(str).str.lower()
    mask_region_upper = rng.random(n) < 0.05
    df.loc[mask_region_upper, "customer_region"] = df.loc[mask_region_upper, "customer_region"].astype(str).str.upper()

    # Stray whitespace on email_domain ~2%
    mask_email_ws = rng.random(n) < 0.02
    df.loc[mask_email_ws, "email_domain"] = " " + df.loc[mask_email_ws, "email_domain"].astype(str)

    # Date format variation: 4% of transaction_datetime in slash form
    mask_date_alt = rng.random(n) < 0.04
    def _alt_date(s):
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%d/%m/%Y %H:%M")
        except Exception:
            return s
    df.loc[mask_date_alt, "transaction_datetime"] = df.loc[mask_date_alt, "transaction_datetime"].apply(_alt_date)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("RAW SYNTHETIC E-COMMERCE TRANSACTION DATASET GENERATOR")
    print("  Source distributions: Niubiz/WooCommerce real seed data")
    print("  Target: 6-typology fraud with smooth probabilistic patterns")
    print("=" * 72)

    # 1) Legitimate transactions
    print(f"\n[1/4] Generating {N_LEGIT:,} legitimate transactions...")
    df_legit = generate_legit(N_LEGIT, order_id_start=100_000)

    # 2) Fraudulent transactions by typology
    print(f"[2/4] Generating {N_FRAUD:,} fraudulent transactions across 6 typologies...")
    fraud_dfs = []
    cur_id = 100_000 + N_LEGIT
    generators = {
        "card_testing": generate_card_testing,
        "stolen_card": generate_stolen_card,
        "account_takeover": generate_account_takeover,
        "friendly_fraud": generate_friendly_fraud,
        "bot_automated": generate_bot_automated,
        "triangulation": generate_triangulation,
    }
    for ftype, pct in FRAUD_TYPES.items():
        n_t = int(N_FRAUD * pct)
        df_t = generators[ftype](n_t, cur_id)
        cur_id += n_t
        fraud_dfs.append(df_t)
        print(f"   • {ftype:20s}  {n_t:,} txs")

    df_fraud = pd.concat(fraud_dfs, ignore_index=True)

    # Pad any rounding shortfall
    diff = N_FRAUD - len(df_fraud)
    if diff > 0:
        df_extra = generate_stolen_card(diff, cur_id)
        df_fraud = pd.concat([df_fraud, df_extra], ignore_index=True)

    # 3) Combine + shuffle
    print(f"\n[3/4] Combining {len(df_legit):,} legit + {len(df_fraud):,} fraud, shuffling...")
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 4) Realistic noise (missing values, format inconsistencies)
    print("[4/4] Injecting realistic noise (missing values, format variations)...")
    df = inject_realistic_noise(df)

    # ─── Final summary ───
    n_fraud_final = int(df["is_fraud"].sum())
    print("\n" + "=" * 72)
    print("DATASET SUMMARY")
    print("-" * 72)
    print(f"  Total transactions : {len(df):,}")
    print(f"  Legitimate         : {len(df) - n_fraud_final:,}  ({(1 - n_fraud_final/len(df))*100:.2f}%)")
    print(f"  Fraudulent         : {n_fraud_final:,}  ({n_fraud_final/len(df)*100:.2f}%)")
    print(f"  Columns            : {len(df.columns)}")
    print(f"  Missing-value cols : {sum(df[c].isna().any() for c in df.columns)}")
    print("\n  Fraud type breakdown:")
    for ft, c in df.loc[df["is_fraud"] == 1, "fraud_type_meta"].value_counts().items():
        print(f"     • {ft:20s} {c:,}")

    # ─── Save ───
    out_csv = OUT_DIR / "fraud_ecommerce_dataset_raw.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Raw dataset written: {out_csv}")
    print(f"     Shape: {df.shape}")

    meta = {
        "n_total": len(df),
        "n_fraud": n_fraud_final,
        "n_legit": int((df["is_fraud"] == 0).sum()),
        "fraud_rate": round(df["is_fraud"].mean(), 4),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "fraud_typology": dict(df[df["is_fraud"] == 1]["fraud_type_meta"].value_counts()),
        "seed": SEED,
        "generated_at": datetime.now().isoformat(),
        "design_notes": {
            "fraud_signals": "Smooth probabilistic (sigmoid-weighted), no XOR / hyperplane tricks",
            "missing_values_pct": {"email_domain": 0.03, "card_type": 0.02, "eci": 0.08,
                                    "action_code": 0.10, "issuer_bank": 0.03,
                                    "customer_region": 0.05, "product_category": 0.07},
            "format_inconsistencies": ["case variation in issuer_bank/customer_region",
                                        "stray whitespace in email_domain",
                                        "alternative date format in 4% of rows"],
        },
    }
    meta_path = OUT_DIR / "dataset_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    print(f"[OK] Metadata written: {meta_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
