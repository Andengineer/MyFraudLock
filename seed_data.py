#!/usr/bin/env python3
"""
seed_data.py — Limpia y genera datos simulados para MyFraudLock (DAFD-Net).
Ejecutar: python seed_data.py
"""
import os, sys, django, random, json
from datetime import timedelta
from decimal import Decimal

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MyFraudLock.settings")
django.setup()

from django.utils import timezone
from api.models import Transaccion, Incidente, Configuracion, Usuario

# ── Limpiar todo ─────────────────────────────────────────────────────
print("🗑️  Eliminando incidentes y transacciones existentes...")
Incidente.objects.all().delete()
Transaccion.objects.all().delete()
print("   ✓ Base de datos limpia")

# ── Asegurar configuración ───────────────────────────────────────────
config, _ = Configuracion.objects.get_or_create(id=1, defaults={"umbral_score": 70})

# ── Datos base ───────────────────────────────────────────────────────
BRANDS = ["visa", "mastercard", "amex", "diners"]
TYPES = ["credito", "debito"]
BANKS = ["bcp", "bbva", "interbank", "scotiabank", "falabella", "ripley", "banbif", "otros"]
CHANNELS = ["pago web", "pago movil", "app"]
REGIONS = ["lima", "arequipa", "piura", "cusco", "lambayeque", "la_libertad", "junin", "callao"]
CATEGORIES = ["repuestos_moto", "indumentaria_moto", "aceites_lubricantes", "cascos", "accesorios", "electronica", "otros"]
EMAILS_LEGIT = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "icloud.com"]
EMAILS_FRAUD = ["yopmail.com", "mailinator.com", "guerrillamail.com", "tempmail.com", "10minutemail.com"]
DENIAL_REASONS = ["unknown", "denegada", "limite excedido", "tarjeta vencida"]
BINS_LEGIT = ["404700", "454620", "520000", "370000", "362500"]
BINS_FRAUD = ["400000", "411111", "999999", "000000"]

now = timezone.now()

def rand_date(days_back=30):
    return now - timedelta(days=random.uniform(0, days_back), hours=random.randint(0, 23), minutes=random.randint(0, 59))


def make_legit():
    """Transacción legítima típica."""
    return {
        "importe": Decimal(str(round(random.uniform(15, 350), 2))),
        "card_brand": random.choice(BRANDS),
        "card_type": random.choice(TYPES),
        "issuer_bank": random.choice(BANKS),
        "payment_channel": random.choice(CHANNELS),
        "eci_code": random.choice([2, 5]),  # Con 3DS
        "num_installments": random.choice([0, 0, 0, 3, 6]),
        "customer_region": random.choice(REGIONS),
        "product_category": random.choice(CATEGORIES),
        "num_items": random.randint(1, 4),
        "discount_amount": Decimal(str(round(random.uniform(0, 20), 2))),
        "currency": "PEN",
        "transaction_status": "liquidada",
        "action_code": 6,
        "denial_reason": "unknown",
        "email_domain": random.choice(EMAILS_LEGIT),
        "bin": random.choice(BINS_LEGIT),
        "wallet_yape": random.choice(["no", "no", "si"]),
        "wallet_plin": random.choice(["no", "no", "si"]),
        "ratio_aar": Decimal(str(round(random.uniform(0.5, 1.5), 4))),
        "ratio_cmr": Decimal(str(round(random.uniform(0.5, 1.5), 4))),
        "ratio_asi": Decimal(str(round(random.uniform(0.4, 0.9), 4))),
        "ratio_vrr": Decimal(str(round(random.uniform(0.5, 3.0), 4))),
        "ratio_dar": Decimal(str(round(random.uniform(0.0, 0.1), 4))),
        "ratio_csi": Decimal("1.0000"),
        "ratio_dpe": Decimal(str(round(random.uniform(0.0, 0.1), 4))),
    }


def make_fraud():
    """Transacción fraudulenta típica."""
    return {
        "importe": Decimal(str(round(random.uniform(800, 5000), 2))),
        "card_brand": random.choice(BRANDS),
        "card_type": "credito",
        "issuer_bank": random.choice(BANKS),
        "payment_channel": random.choice(CHANNELS),
        "eci_code": random.choice([0, 6, 7]),  # Sin 3DS
        "num_installments": 0,
        "customer_region": random.choice(REGIONS),
        "product_category": random.choice(["electronica", "accesorios", "otros"]),
        "num_items": 1,
        "discount_amount": Decimal("0.00"),
        "currency": "PEN",
        "transaction_status": random.choice(["denegada", "liquidada", "abandonada"]),
        "action_code": random.choice([0, 1, 2]),
        "denial_reason": random.choice(DENIAL_REASONS[1:]),
        "email_domain": random.choice(EMAILS_FRAUD),
        "bin": random.choice(BINS_FRAUD),
        "wallet_yape": "no",
        "wallet_plin": "no",
        "ratio_aar": Decimal(str(round(random.uniform(3.0, 8.0), 4))),
        "ratio_cmr": Decimal(str(round(random.uniform(2.5, 6.0), 4))),
        "ratio_asi": Decimal(str(round(random.uniform(0.0, 0.2), 4))),
        "ratio_vrr": Decimal(str(round(random.uniform(15.0, 60.0), 4))),
        "ratio_dar": Decimal(str(round(random.uniform(0.3, 0.9), 4))),
        "ratio_csi": Decimal(str(round(random.uniform(2.0, 5.0), 4))),
        "ratio_dpe": Decimal(str(round(random.uniform(0.4, 1.0), 4))),
    }


# ── Generar transacciones ────────────────────────────────────────────
print("\n📊 Generando transacciones simuladas...")

created_tx = []
# 25 legítimas (bajo score, no generan incidente)
for i in range(25):
    data = make_legit()
    tx = Transaccion(**data)
    tx.save()
    # Override fecha (auto_now_add)
    Transaccion.objects.filter(pk=tx.pk).update(fecha=rand_date(30))
    created_tx.append(("legit", tx))

# 20 fraudulentas (alto score, generan incidente)
for i in range(20):
    data = make_fraud()
    tx = Transaccion(**data)
    tx.save()
    Transaccion.objects.filter(pk=tx.pk).update(fecha=rand_date(30))
    created_tx.append(("fraud", tx))

print(f"   ✓ {len(created_tx)} transacciones creadas (25 legítimas + 20 fraudulentas)")

# ── Generar incidentes para las fraudulentas ─────────────────────────
print("\n🚨 Generando incidentes...")

estados_fraud = ["Fraude confirmado"] * 12 + ["Pendiente"] * 5 + ["Falso positivo"] * 3
random.shuffle(estados_fraud)

# Analistas/Admins que pueden gestionar incidentes (para trazabilidad).
gestores = list(Usuario.objects.filter(rol__in=["ANALISTA", "ADMIN"], activo=True))

fraud_txs = [t for kind, t in created_tx if kind == "fraud"]
for i, tx in enumerate(fraud_txs):
    tx.refresh_from_db()
    score = round(random.uniform(72, 99), 2)
    estado = estados_fraud[i] if i < len(estados_fraud) else "Pendiente"

    # Explicabilidad simulada (formato real del sistema)
    top_factors = []
    factor_pool = [
        ("DAR", "Ratio Denegación/Intentos (DAR)", round(random.uniform(0.05, 0.3), 4)),
        ("VRR", "Ratio Velocidad Riesgo (VRR)", round(random.uniform(0.03, 0.2), 4)),
        ("AAR", "Ratio Monto/Promedio (AAR)", round(random.uniform(0.02, 0.15), 4)),
        ("email_domain", "Dominio Email", round(random.uniform(0.02, 0.1), 4)),
        ("eci", "Código ECI (Autenticación)", round(random.uniform(0.01, 0.08), 4)),
        ("transaction_amount", "Monto Transacción", round(random.uniform(0.01, 0.12), 4)),
    ]
    random.shuffle(factor_pool)
    for feat, display, impact in factor_pool[:4]:
        top_factors.append({"feature": feat, "display_name": display, "impact": impact, "value": "—"})

    explicacion_negocio = []
    if score > 85:
        explicacion_negocio.append("Alto ratio de denegaciones: la tarjeta tiene historial de intentos rechazados.")
        explicacion_negocio.append("Velocidad de transacciones anormalmente alta, compatible con card testing.")
    if score > 75:
        explicacion_negocio.append("El monto supera significativamente el promedio del cliente.")
    if tx.email_domain in EMAILS_FRAUD:
        explicacion_negocio.append(f"Email usa dominio temporal/desechable ({tx.email_domain}).")

    explicabilidad = {
        "top_factors": top_factors,
        "explicacion_negocio": explicacion_negocio,
        "prob": round(score / 100, 6),
        "sum_abs": round(sum(f["impact"] for f in top_factors), 4),
    }

    # Un incidente resuelto (confirmado / falso positivo) siempre debe registrar
    # qué analista lo gestionó. Los pendientes quedan sin gestor.
    gestor = random.choice(gestores) if (gestores and estado != "Pendiente") else None

    inc = Incidente.objects.create(
        id_transaccion=tx,
        score_riesgo=Decimal(str(score)),
        explicabilidad=explicabilidad,
        estado=estado,
        gestionado_por=gestor,
        comentario="Evaluación automática del sistema DAFD-Net" if estado != "Pendiente" else "",
    )
    # Override fecha
    Incidente.objects.filter(pk=inc.pk).update(fecha=tx.fecha + timedelta(seconds=random.randint(1, 60)))

estados = {}
for inc in Incidente.objects.all():
    estados[inc.estado] = estados.get(inc.estado, 0) + 1

print(f"   ✓ {Incidente.objects.count()} incidentes creados")
for est, cnt in estados.items():
    print(f"     • {est}: {cnt}")

print(f"\n✅ Seed completado exitosamente")
print(f"   Transacciones: {Transaccion.objects.count()}")
print(f"   Incidentes:    {Incidente.objects.count()}")
