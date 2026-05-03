import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'MyFraudLock.settings')
django.setup()

from api.models import Incidente
from api.pdf_utils import build_dashboard_pdf, build_incidentes_pdf, build_incidente_detalle_pdf

ctx = {"pendientes": 10, "confirmados": 5, "falsos": 3, "recientes": list(Incidente.objects.all()[:5])}
try:
    build_dashboard_pdf(ctx)
    print("Dashboard PDF OK")
except Exception as e:
    print("Dashboard PDF error:", e)

try:
    build_incidentes_pdf(Incidente.objects.all()[:5])
    print("Incidentes PDF OK")
except Exception as e:
    print("Incidentes PDF error:", e)

try:
    build_incidente_detalle_pdf(Incidente.objects.first())
    print("Detalle PDF OK")
except Exception as e:
    print("Detalle PDF error:", e)
