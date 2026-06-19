#!/usr/bin/env python3
"""
seed_data.py — Limpia y genera datos sintéticos para MyFraudLock (modelo FraudDNN).

- Borra transacciones e incidentes existentes (no toca los usuarios reales).
- Asegura usuarios gestores (analistas/admin) para la trazabilidad.
- Genera transacciones legítimas y fraudulentas con el vocabulario del modelo.
- Calcula el score y la explicabilidad con el MODELO REAL (FraudDNN + SHAP).
- Crea incidentes para las transacciones que superan el umbral, registrando
  estado, comentario y el usuario que las gestionó.

Ejecutar:  python seed_data.py
"""
import os, sys, django, random, json
from datetime import timedelta
from decimal import Decimal

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MyFraudLock.settings")
django.setup()

from django.utils import timezone
from api.models import Transaccion, Incidente, Configuracion, Usuario
from api.ml_utils import predict_fraud
from api.ml import xai

# SHAP más liviano durante el sembrado (explicaciones igual de representativas)
xai.SHAP_NSAMPLES = 50

# ── Limpiar transacciones e incidentes (no usuarios) ─────────────────
print("Eliminando incidentes y transacciones existentes...")
Incidente.objects.all().delete()
Transaccion.objects.all().delete()
print("   OK base limpia")

# ── Configuración ────────────────────────────────────────────────────
config, _ = Configuracion.objects.get_or_create(id=1, defaults={"umbral_score": 70})
UMBRAL = config.umbral_score

# ── Asegurar usuarios gestores (no resetea los existentes) ───────────
DEMO_USERS = [
    ("admin",      "admin@myfraudlock.pe",      "ADMIN"),
    ("ana.torres", "ana.torres@myfraudlock.pe", "ANALISTA"),
    ("luis.rojas", "luis.rojas@myfraudlock.pe", "ANALISTA"),
    ("gerente",    "gerente@myfraudlock.pe",    "GERENTE"),
]
DEMO_PASSWORD = "MyFraud2026*"
for uname, mail, rol in DEMO_USERS:
    u, created = Usuario.objects.get_or_create(
        username=uname, defaults={"email": mail, "rol": rol, "activo": True}
    )
    if created:
        u.set_password(DEMO_PASSWORD)
        u.save()
        print(f"   + usuario demo creado: {uname} ({rol})  contraseña: {DEMO_PASSWORD}")

gestores = list(Usuario.objects.filter(rol__in=["ANALISTA", "ADMIN"], activo=True))

# ── Vocabulario (coincide con el del modelo) ─────────────────────────
BRANDS = ["visa", "mastercard", "amex", "diners"]
TYPES = ["credito", "debito"]
BANKS = ["bcp", "bbva", "interbank", "scotiabank", "falabella", "ripley", "banbif"]
CHANNELS = ["pago web", "pago movil", "app"]
REGIONS = ["lima", "arequipa", "piura", "cusco", "lambayeque", "la_libertad", "junin", "callao"]
CATEGORIES = ["casco", "guantes", "bota", "pantalon", "protector", "aceite", "kit", "otros"]
EMAILS_LEGIT = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "icloud.com"]
EMAILS_FRAUD = ["yopmail.com", "mailinator.com", "guerrillamail.com", "tempmail.com"]
DENIAL_REASONS = ["denegada", "limite excedido", "tarjeta vencida"]
BINS_LEGIT = ["404700", "454620", "520000", "370000", "362500"]
BINS_FRAUD = ["400000", "411111", "999999", "000000"]

now = timezone.now()


def rand_date(days_back=30):
    return now - timedelta(days=random.uniform(0, days_back),
                           hours=random.randint(0, 23), minutes=random.randint(0, 59))


def r4(lo, hi):
    return round(random.uniform(lo, hi), 4)


def make_legit():
    """Perfil de transacción legítima (ratios benignos)."""
    return {
        "importe": round(random.uniform(15, 350), 2),
        "card_brand": random.choice(BRANDS), "card_type": random.choice(TYPES),
        "issuer_bank": random.choice(BANKS), "payment_channel": random.choice(CHANNELS),
        "eci": random.choice([2, 5]), "num_installments": random.choice([0, 0, 0, 3, 6]),
        "customer_region": random.choice(REGIONS), "product_category": random.choice(CATEGORIES),
        "num_items": random.randint(1, 4), "discount_amount": round(random.uniform(0, 20), 2),
        "currency": "PEN", "transaction_status": "liquidada", "action_code": 6,
        "denial_reason": "unknown", "email_domain": random.choice(EMAILS_LEGIT),
        "bin": random.choice(BINS_LEGIT),
        "wallet_yape": random.choice(["no", "no", "si"]), "wallet_plin": random.choice(["no", "no", "si"]),
        "AAR": r4(0.5, 1.5), "CMR": r4(0.5, 1.5), "ASI": r4(0.6, 0.95), "VRR": r4(0.5, 3.0),
        "DAR": r4(0.0, 0.08), "CSI": 1.0, "DPE": r4(0.0, 0.1),
    }


def make_fraud():
    """Perfil de transacción fraudulenta (ratios de riesgo)."""
    return {
        "importe": round(random.uniform(800, 5000), 2),
        "card_brand": random.choice(BRANDS), "card_type": "credito",
        "issuer_bank": random.choice(BANKS), "payment_channel": random.choice(CHANNELS),
        "eci": random.choice([0, 6, 7]), "num_installments": 0,
        "customer_region": random.choice(REGIONS), "product_category": random.choice(CATEGORIES),
        "num_items": 1, "discount_amount": 0.0,
        "currency": "PEN", "transaction_status": random.choice(["denegada", "liquidada", "abandonada"]),
        "action_code": random.choice([0, 1, 2]), "denial_reason": random.choice(DENIAL_REASONS),
        "email_domain": random.choice(EMAILS_FRAUD), "bin": random.choice(BINS_FRAUD),
        "wallet_yape": "no", "wallet_plin": "no",
        "AAR": r4(3.0, 8.0), "CMR": r4(2.5, 6.0), "ASI": r4(0.0, 0.2), "VRR": r4(15.0, 60.0),
        "DAR": r4(0.3, 0.9), "CSI": r4(2.0, 5.0), "DPE": r4(0.4, 1.0),
    }


def to_tx_kwargs(p):
    """Mapea el payload (dict) a los campos del modelo Transaccion."""
    return dict(
        importe=Decimal(str(p["importe"])),
        card_brand=p["card_brand"], card_type=p["card_type"], issuer_bank=p["issuer_bank"],
        payment_channel=p["payment_channel"], eci_code=p["eci"], num_installments=p["num_installments"],
        bin=p["bin"], customer_region=p["customer_region"], email_domain=p["email_domain"],
        product_category=p["product_category"], num_items=p["num_items"],
        discount_amount=Decimal(str(p["discount_amount"])), currency=p["currency"],
        transaction_status=p["transaction_status"], action_code=p["action_code"],
        denial_reason=p["denial_reason"], wallet_yape=p["wallet_yape"], wallet_plin=p["wallet_plin"],
        ratio_aar=Decimal(str(p["AAR"])), ratio_cmr=Decimal(str(p["CMR"])),
        ratio_asi=Decimal(str(p["ASI"])), ratio_vrr=Decimal(str(p["VRR"])),
        ratio_dar=Decimal(str(p["DAR"])), ratio_csi=Decimal(str(p["CSI"])),
        ratio_dpe=Decimal(str(p["DPE"])),
    )


# ── Generar transacciones + scoring con el modelo real ───────────────
print("\nGenerando transacciones y evaluándolas con FraudDNN (puede tardar)...")
random.seed(42)

profiles = [("legit", make_legit()) for _ in range(30)] + \
           [("fraud", make_fraud()) for _ in range(22)]
random.shuffle(profiles)

scored = []  # (tx, score, explicabilidad_dict)
for kind, payload in profiles:
    score, exp_str = predict_fraud(payload)   # usa los ratios provistos
    try:
        exp = json.loads(exp_str)
    except (json.JSONDecodeError, TypeError):
        exp = {}
    tx = Transaccion(**to_tx_kwargs(payload))
    tx.save()
    Transaccion.objects.filter(pk=tx.pk).update(fecha=rand_date(30))
    tx.refresh_from_db()
    scored.append((tx, round(float(score), 2), exp))

print(f"   OK {len(scored)} transacciones creadas y evaluadas")

# ── Crear incidentes para los que superan el umbral ──────────────────
print(f"\nGenerando incidentes (umbral = {UMBRAL}%)...")
flagged = [(tx, sc, exp) for (tx, sc, exp) in scored if sc >= UMBRAL]

# Distribución de estados: la mayoría gestionados (con usuario), algunos pendientes
n = len(flagged)
estados = (["Fraude confirmado"] * round(n * 0.55) +
           ["Falso positivo"] * round(n * 0.15) +
           ["Pendiente"] * n)[:n]
random.shuffle(estados)

COMENTARIOS_FRAUDE = [
    "Patrón de card testing confirmado: alto DAR y velocidad anómala.",
    "Monto muy por encima de la mediana de la categoría; tarjeta sin 3D Secure.",
    "Comportamiento consistente con apropiación de cuenta; bloqueado.",
]
COMENTARIOS_FP = [
    "Cliente recurrente verificado por teléfono; transacción legítima.",
    "Compra alta pero coherente con el historial del cliente; liberada.",
]

for (tx, sc, exp), estado in zip(flagged, estados):
    gestor = random.choice(gestores) if (gestores and estado != "Pendiente") else None
    if estado == "Fraude confirmado":
        comentario = random.choice(COMENTARIOS_FRAUDE)
    elif estado == "Falso positivo":
        comentario = random.choice(COMENTARIOS_FP)
    else:
        comentario = ""
    inc = Incidente.objects.create(
        id_transaccion=tx,
        score_riesgo=Decimal(str(sc)),
        explicabilidad=exp,
        estado=estado,
        gestionado_por=gestor,
        comentario=comentario,
    )
    Incidente.objects.filter(pk=inc.pk).update(
        fecha=tx.fecha + timedelta(seconds=random.randint(1, 90))
    )

# ── Resumen ──────────────────────────────────────────────────────────
resumen = {}
for inc in Incidente.objects.all():
    resumen[inc.estado] = resumen.get(inc.estado, 0) + 1

print(f"   OK {Incidente.objects.count()} incidentes creados")
for est, cnt in resumen.items():
    print(f"     - {est}: {cnt}")
gestionados = Incidente.objects.exclude(gestionado_por=None).count()
print(f"     - con usuario gestor: {gestionados}")

print("\nSeed completado.")
print(f"   Transacciones: {Transaccion.objects.count()}")
print(f"   Incidentes:    {Incidente.objects.count()}")
print(f"   Usuarios:      {Usuario.objects.count()}")
