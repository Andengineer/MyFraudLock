# api/ml_utils.py
"""
Utilidades ML para detección de fraude — DAFD-Net.
Calcula ratios financieros en tiempo real y ejecuta predicción.
"""
import json
from math import log2
from decimal import Decimal

from django.utils import timezone
from django.db.models import Avg, Count, Q

from .ml.xai import predict_and_explain


# ─── Helpers ─────────────────────────────────────────────────────────

def _derive_time_parts(fecha):
    """Deriva features temporales usando tz local."""
    if not fecha:
        fecha = timezone.now()
    local = timezone.localtime(fecha)
    return local.hour, local.weekday(), local.month


def _shannon_entropy(values):
    """Shannon entropy en bits para una lista de valores categóricos."""
    if not values:
        return 0.0
    total = len(values)
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return float(-sum((c / total) * log2(c / total) for c in counts.values() if c > 0))


def _compute_ratios(tx_amount, bin_val, product_category, eci, last_4_digits="0000"):
    """
    Calcula los 7 ratios financieros en tiempo real usando el historial
    de transacciones en la base de datos.

    Replica la lógica de Entrenamiento/02b_preprocess_and_label.py §3.C.2
    """
    from .models import Transaccion  # import local para evitar circular

    now = timezone.now()
    last_24h = now - timezone.timedelta(hours=24)

    card_full_id = f"{bin_val}{last_4_digits}"

    # ── AAR: Amount-to-Average Ratio ─────────────────────────────────
    # tx_amount / promedio histórico del mismo BIN
    avg_result = Transaccion.objects.filter(bin=bin_val).aggregate(
        avg_amt=Avg("importe")
    )
    avg_amt = float(avg_result["avg_amt"] or tx_amount)  # si no hay historial, AAR=1
    AAR = round(tx_amount / max(avg_amt, 0.001), 4)
    AAR = min(AAR, 50.0)

    # ── CMR: Category Median Ratio ───────────────────────────────────
    # tx_amount / mediana de la categoría
    cat_amounts = list(
        Transaccion.objects.filter(product_category=product_category)
        .values_list("importe", flat=True)
    )
    if cat_amounts:
        cat_amounts_sorted = sorted(float(a) for a in cat_amounts)
        n = len(cat_amounts_sorted)
        median_amt = (cat_amounts_sorted[n // 2] if n % 2 == 1
                      else (cat_amounts_sorted[n // 2 - 1] + cat_amounts_sorted[n // 2]) / 2)
    else:
        median_amt = tx_amount
    CMR = round(tx_amount / max(median_amt, 0.001), 4)
    CMR = min(CMR, 50.0)

    # ── ASI: Authentication Strength Index ───────────────────────────
    # Fracción de transacciones del BIN que usaron 3DS fuerte (ECI 2 o 5)
    bin_txs = Transaccion.objects.filter(bin=bin_val)
    total_bin = bin_txs.count()
    if total_bin > 0:
        strong_auth_count = bin_txs.filter(eci_code__in=[2, 5]).count()
        # Incluir la transacción actual
        is_current_strong = 1 if eci in [2, 5] else 0
        ASI = round((strong_auth_count + is_current_strong) / (total_bin + 1), 4)
    else:
        ASI = 1.0 if eci in [2, 5] else 0.0

    # ── VRR: Velocity Risk Ratio ─────────────────────────────────────
    # Transacciones del BIN en últimas 24h / promedio diario del BIN
    count_24h = Transaccion.objects.filter(
        bin=bin_val, fecha__gte=last_24h
    ).count() + 1  # +1 para la actual

    # Promedio diario: total de tx / días observados
    first_tx = Transaccion.objects.filter(bin=bin_val).order_by("fecha").first()
    if first_tx and total_bin > 0:
        days_observed = max(1, (now - first_tx.fecha).days + 1)
        avg_daily = total_bin / days_observed
        VRR = round(count_24h / max(avg_daily, 0.001), 4)
    else:
        VRR = 1.0  # primera transacción del BIN
    VRR = min(VRR, 50.0)

    # ── DAR: Denial-to-Attempt Ratio ─────────────────────────────────
    # Denegaciones en últimas 24h / total intentos en 24h para esta tarjeta
    attempts_24h = Transaccion.objects.filter(
        bin=bin_val, fecha__gte=last_24h
    )
    total_attempts = attempts_24h.count() + 1  # +1 actual
    denials_24h = attempts_24h.filter(
        transaction_status="denegada"
    ).count()
    DAR = round(denials_24h / max(total_attempts, 1), 4)
    DAR = min(DAR, 1.0)

    # ── CSI: Card Sharing Index ──────────────────────────────────────
    # Número de regiones distintas que han usado esta tarjeta (BIN)
    distinct_regions = Transaccion.objects.filter(
        bin=bin_val
    ).values("customer_region").distinct().count()
    CSI = max(distinct_regions, 1)

    # ── DPE: Denial Pattern Entropy ──────────────────────────────────
    # Entropía de Shannon de las razones de denegación para esta tarjeta
    denial_reasons = list(
        Transaccion.objects.filter(
            bin=bin_val,
            transaction_status="denegada"
        ).exclude(
            denial_reason__in=["unknown", ""]
        ).values_list("denial_reason", flat=True)
    )
    DPE = round(_shannon_entropy(denial_reasons), 4)

    return {
        "AAR": AAR, "CMR": CMR, "ASI": ASI,
        "VRR": VRR, "DAR": DAR, "CSI": float(CSI), "DPE": DPE,
    }


# ─── Predicción principal ────────────────────────────────────────────

def predict_fraud(tx):
    """
    tx: instancia Transaccion o dict.

    Calcula automáticamente:
      - Features temporales (hora, día, mes)
      - 7 ratios financieros (AAR, CMR, ASI, VRR, DAR, CSI, DPE)
    """
    get = (lambda k: getattr(tx, k, None)) if not isinstance(tx, dict) else (lambda k: tx.get(k))

    # ── Temporales ──────────────────────────────────────────────────
    fecha = get("fecha")
    tx_hour, tx_day_of_week, tx_month = _derive_time_parts(fecha)

    # ── Monto ───────────────────────────────────────────────────────
    amt = float(get("importe") or get("transaction_amount") or 0.0)
    discount = float(get("discount_amount") or 0.0)

    # ── ECI / Autenticación ─────────────────────────────────────────
    eci = int(get("eci") or get("eci_code") or 5)
    action_code = int(get("action_code") or 6)

    # ── Cuotas / Items ──────────────────────────────────────────────
    num_installments = int(get("num_installments") or 0)
    num_items = int(get("num_items") or 1)

    # ── Categóricas ─────────────────────────────────────────────────
    card_brand       = (get("card_brand") or "visa").strip().lower()
    card_type        = (get("card_type") or "credito").strip().lower()
    currency         = (get("currency") or "PEN").strip().upper()
    transaction_status = (get("transaction_status") or "liquidada").strip().lower()
    payment_channel  = (get("payment_channel") or "pago web").strip().lower()
    wallet_yape      = (get("wallet_yape") or "no").strip().lower()
    wallet_plin      = (get("wallet_plin") or "no").strip().lower()
    product_category = (get("product_category") or get("category") or "otros").strip().lower()
    issuer_bank      = (get("issuer_bank") or "bcp").strip().lower()
    customer_region  = (get("customer_region") or "lima").strip().lower()
    email_domain     = (get("email_domain") or "gmail.com").strip().lower()
    denial_reason    = (get("denial_reason") or "unknown").strip().lower()
    bin_val          = str(get("bin") or "404700").strip()
    last_4           = str(get("last_4_digits") or "0000").strip()

    # ── Ratios financieros (calculados automáticamente) ─────────────
    ratios = _compute_ratios(amt, bin_val, product_category, eci, last_4)

    payload = {
        # Numéricas
        "transaction_amount": amt,
        "discount_amount":    discount,
        "tx_hour":            tx_hour,
        "tx_day_of_week":     tx_day_of_week,
        "tx_month":           tx_month,
        "eci":                eci,
        "action_code":        action_code,
        "num_installments":   num_installments,
        "num_items":          num_items,
        # Ratios calculados
        "AAR":                ratios["AAR"],
        "CMR":                ratios["CMR"],
        "ASI":                ratios["ASI"],
        "VRR":                ratios["VRR"],
        "DAR":                ratios["DAR"],
        "CSI":                ratios["CSI"],
        "DPE":                ratios["DPE"],
        # Categóricas
        "card_brand":         card_brand,
        "card_type":          card_type,
        "currency":           currency,
        "transaction_status": transaction_status,
        "payment_channel":    payment_channel,
        "wallet_yape":        wallet_yape,
        "wallet_plin":        wallet_plin,
        "product_category":   product_category,
        "issuer_bank":        issuer_bank,
        "customer_region":    customer_region,
        "email_domain":       email_domain,
        "denial_reason":      denial_reason,
        "bin":                bin_val,
    }

    score, exp = predict_and_explain(payload)

    # Guardar ratios calculados de vuelta en la transacción si es un modelo
    if not isinstance(tx, dict) and hasattr(tx, 'pk') and tx.pk:
        from .models import Transaccion as TxModel
        TxModel.objects.filter(pk=tx.pk).update(
            ratio_aar=Decimal(str(ratios["AAR"])),
            ratio_cmr=Decimal(str(ratios["CMR"])),
            ratio_asi=Decimal(str(ratios["ASI"])),
            ratio_vrr=Decimal(str(ratios["VRR"])),
            ratio_dar=Decimal(str(ratios["DAR"])),
            ratio_csi=Decimal(str(ratios["CSI"])),
            ratio_dpe=Decimal(str(ratios["DPE"])),
        )

    return score, json.dumps(exp, ensure_ascii=False)
