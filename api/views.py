"""
Vistas HTML y API ViewSets para MyFraudLock.
"""
import csv
import io
import json
import logging

logger = logging.getLogger(__name__)

from django.contrib import messages
from django.db.models import Count, Sum
from django.db.models.functions import TruncDate
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from .decorators import require_roles, usuario_login_required
from .ml_utils import predict_fraud
from .models import Configuracion, Incidente, Transaccion, Usuario
from .notifications import notify_fraud_confirmed, notify_new_incident
from .pdf_utils import (
    build_dashboard_pdf,
    build_incidente_detalle_pdf,
    build_incidentes_pdf,
)
from .serializers import (
    ConfiguracionSerializer,
    IncidenteSerializer,
    TransaccionSerializer,
    TransaccionReadSerializer,
    UsuarioSerializer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROLE_LABELS = {
    "ADMIN": "Administrador",
    "ANALISTA": "Analista de Incidentes",
    "GERENTE": "Gerente (solo lectura)",
}


def _allowed_roles_for_request(request):
    """Devuelve los roles que el usuario actual puede asignar."""
    is_admin = request.session.get('usuario_rol') == 'ADMIN'
    roles = ["ADMIN", "ANALISTA", "GERENTE"] if is_admin else ["ANALISTA", "GERENTE"]
    return [{"value": r, "label": ROLE_LABELS.get(r, r)} for r in roles]


def validar_telefono(telefono):
    """Valida un número telefónico opcional.

    Acepta vacío (el teléfono no es obligatorio). Si se ingresa, debe contener
    únicamente dígitos (se permiten + inicial, espacios y guiones como separadores
    de formato) y tener entre 7 y 15 dígitos.

    Devuelve (telefono_normalizado, error). Si error es None la validación pasó.
    """
    if not telefono:
        return None, None

    # Quitar separadores de formato comunes
    limpio = telefono.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    if limpio.startswith("+"):
        limpio = limpio[1:]

    if not limpio.isdigit():
        return telefono, "El teléfono solo puede contener números."

    if not (7 <= len(limpio) <= 15):
        return telefono, "El teléfono debe tener entre 7 y 15 dígitos."

    return telefono, None


# ---------------------------------------------------------------------------
# API REST — ViewSets
# ---------------------------------------------------------------------------

class UsuarioViewSet(viewsets.ModelViewSet):
    """CRUD REST para usuarios."""
    queryset = Usuario.objects.all()
    serializer_class = UsuarioSerializer


class TransaccionViewSet(viewsets.ModelViewSet):
    """CRUD REST para transacciones — ejecuta predicción al crear."""
    queryset = Transaccion.objects.all()
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['product_category', 'card_brand', 'customer_region']

    def get_serializer_class(self):
        if self.action in ('list', 'retrieve'):
            return TransaccionReadSerializer
        return TransaccionSerializer

    def create(self, request, *args, **kwargs):
        serializer = TransaccionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        transaccion = serializer.save()

        try:
            score, explicabilidad_str = predict_fraud(transaccion)
            # explicabilidad viene como JSON string; el JSONField necesita dict
            try:
                explicabilidad_dict = json.loads(explicabilidad_str)
            except (json.JSONDecodeError, TypeError):
                explicabilidad_dict = {"error": "No se pudo procesar la explicabilidad"}
        except Exception as e:
            logger.error("Error en predict_fraud (API create): %s", e, exc_info=True)
            # Fallback: guardar la transacción sin predicción
            transaccion.refresh_from_db()
            resp = TransaccionReadSerializer(transaccion).data
            resp["score_riesgo"] = None
            resp["incidente_generado"] = False
            resp["explicabilidad"] = {"error": f"Error en predicción: {str(e)}"}
            resp["warning"] = "La transacción fue guardada pero la predicción falló."
            return Response(resp, status=status.HTTP_201_CREATED)

        config, _ = Configuracion.objects.get_or_create(
            id=1, defaults={"umbral_score": 70}
        )
        incidente_creado = False
        if score >= config.umbral_score:
            incidente = Incidente.objects.create(
                id_transaccion=transaccion,
                score_riesgo=score,
                explicabilidad=explicabilidad_dict,
                estado="Pendiente",
            )
            notify_new_incident(incidente)
            incidente_creado = True

        # Re-leer la transacción para incluir los ratios guardados
        transaccion.refresh_from_db()
        resp = TransaccionReadSerializer(transaccion).data
        resp["score_riesgo"] = round(score, 2)
        resp["incidente_generado"] = incidente_creado
        resp["explicabilidad"] = explicabilidad_dict
        headers = self.get_success_headers(resp)
        return Response(
            resp, status=status.HTTP_201_CREATED, headers=headers
        )


class IncidenteViewSet(viewsets.ModelViewSet):
    """CRUD REST para incidentes con acciones extra."""
    queryset = Incidente.objects.all().order_by('-fecha')
    serializer_class = IncidenteSerializer
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['estado']
    ordering_fields = ['score_riesgo', 'fecha']
    ordering = ['-fecha']

    @action(detail=True, methods=['patch'])
    def cambiar_estado(self, request, pk=None):
        """Cambia estado de un incidente (Fraude confirmado / Falso positivo)."""
        incidente = self.get_object()
        nuevo_estado = request.data.get("estado")

        if nuevo_estado not in ("Fraude confirmado", "Falso positivo"):
            return Response(
                {"error": "Estado no válido. Usa 'Fraude confirmado' o 'Falso positivo'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        incidente.estado = nuevo_estado
        incidente.comentario = request.data.get("comentario", "")
        incidente.save()
        return Response(
            {"mensaje": f"Incidente {incidente.id_incidente} actualizado a {incidente.estado}."},
            status=status.HTTP_200_OK,
        )

    @action(detail=False, methods=['get'])
    def since(self, request):
        """Devuelve incidentes con id > after_id (máx 50) — para polling."""
        try:
            after = int(request.query_params.get('after_id', 0))
        except (TypeError, ValueError):
            after = 0

        qs = Incidente.objects.filter(id_incidente__gt=after).order_by('-fecha')[:50]
        items = [
            {
                "id_incidente": i.id_incidente,
                "score_riesgo": float(i.score_riesgo or 0),
                "estado": i.estado,
                "fecha": i.fecha.isoformat(),
            }
            for i in qs
        ]
        return Response({"items": items})


class SimulacionView(APIView):
    """Simulación de riesgo temporal vía API REST (no persiste)."""

    def post(self, request):
        data = request.data
        if isinstance(data, dict):
            data = [data]

        resultados = []
        for transaccion in data:
            try:
                score, explicabilidad = predict_fraud(transaccion)
                resultados.append({
                    "transaccion": transaccion,
                    "score_riesgo": score,
                    "explicabilidad": explicabilidad,
                    "mensaje": "Predicción temporal, no guardada en BD.",
                })
            except Exception as e:
                logger.error("Error en predict_fraud (SimulacionView): %s", e, exc_info=True)
                resultados.append({
                    "transaccion": transaccion,
                    "score_riesgo": None,
                    "explicabilidad": {"error": f"Error en predicción: {str(e)}"},
                    "mensaje": "Error al procesar la predicción.",
                })
        return Response(resultados, status=status.HTTP_200_OK)


class ConfiguracionViewSet(viewsets.ModelViewSet):
    """CRUD REST para la configuración global."""
    queryset = Configuracion.objects.all()
    serializer_class = ConfiguracionSerializer


# ---------------------------------------------------------------------------
# Vistas HTML — Dashboard
# ---------------------------------------------------------------------------

@usuario_login_required
@require_roles('GERENTE')
def dashboard_view(request):
    """Dashboard ejecutivo con KPIs e Impactos Financieros."""
    # 1. KPIs Generales
    pendientes_count = Incidente.objects.filter(estado="Pendiente").count()
    confirmados_qs = Incidente.objects.filter(estado="Fraude confirmado")
    falsos_qs = Incidente.objects.filter(estado="Falso positivo")
    
    confirmados = confirmados_qs.count()
    falsos = falsos_qs.count()
    
    # 2. Financiero (S/) -> Dinero protegido
    # The models are Incidente -> fk -> id_transaccion -> importe
    dinero_protegido = confirmados_qs.aggregate(total=Sum('id_transaccion__importe'))['total'] or 0.0
    dinero_falso_pos = falsos_qs.aggregate(total=Sum('id_transaccion__importe'))['total'] or 0.0
    dinero_en_riesgo = Incidente.objects.filter(estado="Pendiente").aggregate(total=Sum('id_transaccion__importe'))['total'] or 0.0

    recientes = Incidente.objects.order_by("-fecha")[:8]

    # Fraude por Marca de Tarjeta
    qs_brand = (
        Incidente.objects.filter(estado="Fraude confirmado")
        .values("id_transaccion__card_brand")
        .annotate(n=Count("id_incidente"), total_monto=Sum("id_transaccion__importe"))
        .order_by("-n")
    )
    labels_brand = [(r["id_transaccion__card_brand"] or "Desconocida").upper() for r in qs_brand]
    data_brand = [r["n"] for r in qs_brand]
    data_brand_money = [float(r["total_monto"] or 0) for r in qs_brand]

    # Fraude por Canal de Pago
    qs_channel = (
        Incidente.objects.filter(estado="Fraude confirmado")
        .values("id_transaccion__payment_channel")
        .annotate(n=Count("id_incidente"))
        .order_by("-n")
    )
    labels_channel = [(r["id_transaccion__payment_channel"] or "Web").capitalize() for r in qs_channel]
    data_channel = [r["n"] for r in qs_channel]

    # Fraude por Categoría de Producto
    qs_cat = (
        Incidente.objects.filter(estado="Fraude confirmado")
        .values("id_transaccion__product_category")
        .annotate(n=Count("id_incidente"), total_monto=Sum("id_transaccion__importe"))
        .order_by("-n")
    )
    labels_cat = [(r["id_transaccion__product_category"] or "Otros").capitalize() for r in qs_cat]
    data_cat = [r["n"] for r in qs_cat]
    data_cat_money = [float(r["total_monto"] or 0) for r in qs_cat]

    # Fraude por Estado de Transacción
    qs_txstatus = (
        Incidente.objects.filter(estado="Fraude confirmado")
        .values("id_transaccion__transaction_status")
        .annotate(n=Count("id_incidente"))
    )
    labels_txstatus = []
    data_txstatus = []
    for r in qs_txstatus:
        labels_txstatus.append((r["id_transaccion__transaction_status"] or "Desconocido").capitalize())
        data_txstatus.append(r["n"])

    # Fraude por Uso de 3DS
    qs_3ds = (
        Incidente.objects.filter(estado="Fraude confirmado")
        .values("id_transaccion__eci_code")
        .annotate(n=Count("id_incidente"))
    )
    labels_3ds = []
    data_3ds = []
    for r in qs_3ds:
        is_3ds = r["id_transaccion__eci_code"] in [2, 5]
        label = "Con 3D Secure" if is_3ds else "Sin 3D Secure"
        if label in labels_3ds:
            data_3ds[labels_3ds.index(label)] += r["n"]
        else:
            labels_3ds.append(label)
            data_3ds.append(r["n"])

    # Fraude por Banco Emisor (emisor es una feature del modelo FraudDNN)
    qs_issuer = (
        Incidente.objects.filter(estado="Fraude confirmado")
        .values("id_transaccion__issuer_bank")
        .annotate(n=Count("id_incidente"))
        .order_by("-n")[:10]
    )
    labels_issuer = [(r["id_transaccion__issuer_bank"] or "Desconocido").upper() for r in qs_issuer]
    data_issuer = [r["n"] for r in qs_issuer]
            
    # Distribución de score (buckets) 
    buckets = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    bucket_labels = ["0–20", "20–40", "40–60", "60–80", "80–100"]
    bucket_counts = []
    for lo, hi in buckets:
        if hi < 100:
            c = Incidente.objects.filter(score_riesgo__gte=lo, score_riesgo__lt=hi).count()
        else:
            c = Incidente.objects.filter(score_riesgo__gte=lo, score_riesgo__lte=hi).count()
        bucket_counts.append(c)

    # Tendencia últimos 30 días - Cantidad Incidentes
    tz_info = timezone.get_current_timezone()
    since = timezone.now() - timezone.timedelta(days=29)
    daily = (
        Incidente.objects.filter(fecha__date__gte=since.date())
        .annotate(d=TruncDate("fecha", tzinfo=tz_info))
        .values("d", "estado")
        .annotate(n=Count("id_incidente"))
        .order_by("d")
    )
    dates = sorted({row["d"] for row in daily})
    date_labels = [d.strftime("%d/%b") for d in dates]
    estados = ["Pendiente", "Fraude confirmado", "Falso positivo"]
    series = {e: [0] * len(dates) for e in estados}
    index = {d: i for i, d in enumerate(dates)}
    for row in daily:
        series[row["estado"]][index[row["d"]]] = row["n"]

    ctx = {
        "pendientes": pendientes_count,
        "confirmados": confirmados,
        "falsos": falsos,
        "dinero_protegido": round(dinero_protegido, 2),
        "dinero_falso_pos": round(dinero_falso_pos, 2),
        "dinero_en_riesgo": round(dinero_en_riesgo, 2),
        "recientes": recientes,
        "labels_brand": json.dumps(labels_brand),
        "data_brand": json.dumps(data_brand),
        "data_brand_money": json.dumps(data_brand_money),
        "labels_channel": json.dumps(labels_channel),
        "data_channel": json.dumps(data_channel),
        "labels_cat": json.dumps(labels_cat),
        "data_cat": json.dumps(data_cat),
        "data_cat_money": json.dumps(data_cat_money),
        "bucket_labels": json.dumps(bucket_labels),
        "bucket_counts": json.dumps(bucket_counts),
        "date_labels": json.dumps(date_labels),
        "serie_pendiente": json.dumps(series["Pendiente"]),
        "serie_fraude": json.dumps(series["Fraude confirmado"]),
        "serie_fp": json.dumps(series["Falso positivo"]),
        "labels_txstatus": json.dumps(labels_txstatus),
        "data_txstatus": json.dumps(data_txstatus),
        "labels_3ds": json.dumps(labels_3ds),
        "data_3ds": json.dumps(data_3ds),
        "labels_issuer": json.dumps(labels_issuer),
        "data_issuer": json.dumps(data_issuer),
    }
    return render(request, "api/dashboard.html", ctx)


# ---------------------------------------------------------------------------
# Vistas HTML — Incidentes
# ---------------------------------------------------------------------------

@usuario_login_required
@require_roles('ANALISTA')
def incidentes_view(request):
    """Listado completo de incidentes."""
    incidentes = Incidente.objects.all().order_by('-fecha')
    return render(request, "api/incidentes.html", {"incidentes": incidentes})


@usuario_login_required
@require_roles('ANALISTA')
def incidente_detalle_view(request, incidente_id):
    """Detalle y gestión de un incidente individual."""
    incidente = get_object_or_404(Incidente, id_incidente=incidente_id)

    if request.method == "POST":
        nuevo_estado = request.POST.get("estado")
        comentario = request.POST.get("comentario", "")

        if nuevo_estado in ("Fraude confirmado", "Falso positivo"):
            incidente.estado = nuevo_estado
            incidente.comentario = comentario
            
            # Registrar quién gestionó el incidente
            usuario_id = request.session.get('usuario_id')
            if usuario_id:
                incidente.gestionado_por_id = usuario_id
                
            incidente.save()
            if nuevo_estado == "Fraude confirmado":
                notify_fraud_confirmed(incidente)
            messages.success(
                request,
                f"Incidente #{incidente.id_incidente} actualizado a {nuevo_estado}.",
            )
            return redirect("incidentes_listado")

    return render(request, "api/incidente_detalle.html", {"incidente": incidente})


# ---------------------------------------------------------------------------
# Vistas HTML — Simulación de Riesgo
# ---------------------------------------------------------------------------

@usuario_login_required
@csrf_exempt
@require_roles('ANALISTA', 'GERENTE')
def simulacion_view(request):
    """Simulación de riesgo individual — formulario + resultado."""
    contexto = {"active_tab": "individual"}

    if request.method == "POST":
        try:
            importe = float(request.POST.get("importe") or 0)
        except ValueError:
            messages.error(request, "Importe inválido.")
            return render(request, "api/simulacion.html", contexto)

        payload = {
            "importe": importe,
            "fecha": None,
            "card_brand": (request.POST.get("card_brand") or "visa").strip().lower(),
            "card_type": (request.POST.get("card_type") or "credito").strip().lower(),
            "issuer_bank": (request.POST.get("issuer_bank") or "bcp").strip().lower(),
            "payment_channel": (request.POST.get("payment_channel") or "pago web").strip().lower(),
            "eci": int(request.POST.get("eci_code") or 5),
            "num_installments": int(request.POST.get("num_installments") or 0),
            "customer_region": (request.POST.get("customer_region") or "lima").strip().lower(),
            "product_category": (request.POST.get("product_category") or "otros").strip().lower(),
            "num_items": int(request.POST.get("num_items") or 1),
            "discount_amount": float(request.POST.get("discount_amount") or 0),
            "currency": (request.POST.get("currency") or "PEN").strip().upper(),
            "email_domain": (request.POST.get("email_domain") or "gmail.com").strip().lower(),
            "bin": (request.POST.get("bin") or "404700").strip(),
            "wallet_yape": (request.POST.get("wallet_yape") or "no").strip().lower(),
            "wallet_plin": (request.POST.get("wallet_plin") or "no").strip().lower(),
            # transaction_status, action_code y denial_reason se asignan por defecto
        }

        # Ratios de comportamiento opcionales (para simular escenarios de riesgo).
        # Si se envían, predict_fraud los usa; si no, los calcula del historial.
        for rk in ("aar", "cmr", "asi", "vrr", "dar", "csi", "dpe"):
            v = request.POST.get(rk)
            if v not in (None, ""):
                try:
                    payload[rk.upper()] = float(v)
                except ValueError:
                    pass

        try:
            score, explicabilidad = predict_fraud(payload)
            contexto.update({
                "resultado": {"score_riesgo": score, "explicabilidad": explicabilidad},
                "active_tab": "individual",
            })
            messages.success(request, "Simulación de riesgo individual procesada.")
        except Exception as e:
            logger.error("Error en predict_fraud (simulacion_view): %s", e, exc_info=True)
            messages.error(request, f"Error al procesar la simulación: {e}")

    return render(request, "api/simulacion.html", contexto)


@usuario_login_required
@require_roles('ANALISTA', 'GERENTE')
def simulacion_lote_view(request):
    """Simulación de riesgo por lote — carga CSV y procesa cada fila."""
    contexto = {"active_tab": "lote"}

    if request.method == "POST":
        archivo = request.FILES.get("archivo")
        if not archivo:
            messages.error(request, "Debes subir un archivo CSV válido.")
            return render(request, "api/simulacion.html", contexto)

        try:
            data = archivo.read().decode("utf-8", errors="ignore")
            reader = csv.DictReader(io.StringIO(data))

            expected = {"importe", "card_brand", "card_type", "customer_region", "product_category"}
            headers = {h.strip() for h in (reader.fieldnames or [])}
            if not expected.issubset(headers):
                messages.error(
                    request,
                    "Cabeceras inválidas. Se esperan: importe,card_brand,card_type,customer_region,product_category"
                    " (y opcionalmente: issuer_bank,payment_channel,eci_code,email_domain,...)",
                )
                return render(request, "api/simulacion.html", contexto)

            resultados = []
            for row in reader:
                try:
                    importe = float(str(row.get("importe", "")).replace(",", "."))
                except ValueError:
                    continue

                # Utility para leer booleanos desde CSV (puede venir "True", "False", "1", "0", etc.)
                def parse_bool(val):
                    if not val:
                        return False
                    return str(val).strip().lower() in ('1', 'true', 'yes', 't', 'y', 'verdadero')

                payload = {
                    "importe": importe,
                    "fecha": None,
                    "card_brand": (row.get("card_brand") or "visa").strip().lower(),
                    "card_type": (row.get("card_type") or "credito").strip().lower(),
                    "issuer_bank": (row.get("issuer_bank") or "bcp").strip().lower(),
                    "payment_channel": (row.get("payment_channel") or "pago web").strip().lower(),
                    "eci": int(row.get("eci_code") or row.get("eci") or 5),
                    "num_installments": int(row.get("num_installments") or 0),
                    "customer_region": (row.get("customer_region") or "lima").strip().lower(),
                    "product_category": (row.get("product_category") or row.get("category") or "otros").strip().lower(),
                    "num_items": int(row.get("num_items") or 1),
                    "discount_amount": float(row.get("discount_amount") or 0),
                    "currency": (row.get("currency") or "PEN").strip().upper(),
                    "email_domain": (row.get("email_domain") or "gmail.com").strip().lower(),
                    "bin": (row.get("bin") or "404700").strip(),
                    "wallet_yape": (row.get("wallet_yape") or "no").strip().lower(),
                    "wallet_plin": (row.get("wallet_plin") or "no").strip().lower(),
                    # transaction_status, action_code y denial_reason se asignan por defecto
                }

                # Ratios opcionales desde el CSV (si vienen, se usan; si no, se calculan)
                for rk in ("aar", "cmr", "asi", "vrr", "dar", "csi", "dpe"):
                    v = row.get(rk)
                    if v not in (None, ""):
                        try:
                            payload[rk.upper()] = float(v)
                        except ValueError:
                            pass

                try:
                    score, explicabilidad = predict_fraud(payload)
                    resultados.append({
                        **payload,
                        "score_riesgo": score,
                        "explicabilidad": explicabilidad,
                    })
                except Exception as e:
                    logger.error("Error en predict_fraud (simulacion_lote): %s", e, exc_info=True)
                    resultados.append({
                        **payload,
                        "score_riesgo": 0,
                        "explicabilidad": {"error": f"Error en predicción: {str(e)}"},
                        "mensaje_error": "Falló la predicción para esta fila."
                    })


            resultados.sort(key=lambda x: x["score_riesgo"], reverse=True)
            contexto.update({"resultados": resultados, "active_tab": "lote"})
            messages.success(
                request,
                f"Simulación por lote procesada. Filas válidas: {len(resultados)}",
            )
        except Exception as e:
            messages.error(request, f"Error procesando el CSV: {e}")

    return render(request, "api/simulacion.html", contexto)


# ---------------------------------------------------------------------------
# Vistas HTML — Configuración
# ---------------------------------------------------------------------------

@usuario_login_required
@require_roles('GERENTE')
def configuracion_front(request):
    """Configuración del umbral de riesgo."""
    config, _ = Configuracion.objects.get_or_create(id=1)

    if request.method == "POST":
        try:
            nuevo_umbral = int(request.POST.get("umbral_score", 70))
        except (TypeError, ValueError):
            messages.error(request, "Umbral inválido.")
            return redirect("configuracion_front")

        config.umbral_score = nuevo_umbral
        config.notificaciones_email = bool(request.POST.get("notificaciones_email"))
        
        # Asignar usuario de la sesión actual
        usuario_id = request.session.get('usuario_id')
        if usuario_id:
            try:
                usuario = Usuario.objects.get(id_usuario=usuario_id)
                config.actualizado_por = usuario
            except Usuario.DoesNotExist:
                pass
                
        config.save()
        messages.success(request, "Configuración actualizada correctamente.")
        return redirect("configuracion_front")

    return render(
        request,
        "api/configuracion.html",
        {"config": config},
    )


# ---------------------------------------------------------------------------
# Vistas HTML — Páginas públicas
# ---------------------------------------------------------------------------

def inicio_view(request):
    """Página principal con KPIs rápidos."""
    pendientes = Incidente.objects.filter(estado="Pendiente").count()
    confirmados = Incidente.objects.filter(estado="Fraude confirmado").count()
    falsos = Incidente.objects.filter(estado="Falso positivo").count()
    return render(request, "api/inicio.html", {
        "pendientes": pendientes,
        "confirmados": confirmados,
        "falsos": falsos,
    })


def ayuda_view(request):
    """Página de información y ayuda."""
    config, _ = Configuracion.objects.get_or_create(id=1)
    return render(request, "api/ayuda.html", {"config": config})


# ---------------------------------------------------------------------------
# Vistas HTML — Auth
# ---------------------------------------------------------------------------

def login_view(request):
    """Inicio de sesión con username o email."""
    if request.session.get('usuario_id'):
        return redirect(request.GET.get('next') or reverse('inicio'))

    if request.method == "POST":
        username_or_email = (request.POST.get('username') or '').strip()
        password = (request.POST.get('password') or '').strip()
        remember = bool(request.POST.get('remember'))
        next_url = (
            request.POST.get('next')
            or request.GET.get('next')
            or reverse('inicio')
        )

        # Busca por username o email
        usuario = None
        for field in ('username', 'email'):
            try:
                usuario = Usuario.objects.get(**{field: username_or_email}, activo=True)
                break
            except Usuario.DoesNotExist:
                pass

        if not usuario:
            messages.error(request, "Usuario no encontrado o inactivo.")
            return render(request, "api/login.html", {"next": next_url})

        if not usuario.check_password(password):
            messages.error(request, "Contraseña incorrecta.")
            return render(request, "api/login.html", {"next": next_url})

        # Crear sesión
        request.session['usuario_id'] = usuario.id_usuario
        request.session['usuario_rol'] = usuario.rol
        request.session['usuario_name'] = usuario.username
        if remember:
            request.session.set_expiry(60 * 60 * 24 * 14)  # 14 días
        else:
            request.session.set_expiry(0)  # hasta cerrar navegador

        messages.success(request, f"¡Bienvenido, {usuario.username}!")
        return redirect(next_url)

    return render(request, "api/login.html", {"next": request.GET.get('next', '')})


def logout_view(request):
    """Cierra la sesión activa."""
    request.session.flush()
    messages.success(request, "Sesión cerrada.")
    return redirect('login')


def register_view(request):
    """Registro de nuevos usuarios con contraseña hasheada."""
    if request.session.get('usuario_id'):
        return redirect(request.GET.get('next') or reverse('inicio'))

    allowed_roles = _allowed_roles_for_request(request)

    if request.method == "POST":
        username = (request.POST.get('username') or '').strip()
        email = (request.POST.get('email') or '').strip().lower()
        telefono = (request.POST.get('telefono') or '').strip()
        password = (request.POST.get('password') or '').strip()
        password2 = (request.POST.get('password2') or '').strip()
        rol = (request.POST.get('rol') or 'ANALISTA').strip().upper()
        next_url = (
            request.POST.get('next')
            or request.GET.get('next')
            or reverse('inicio')
        )

        # Validar rol
        allowed_values = {r["value"] for r in allowed_roles}
        if rol not in allowed_values:
            rol = "ANALISTA"

        # Contexto para re-render en caso de error
        form_ctx = {
            "next": next_url,
            "roles": allowed_roles,
            "selected_rol": rol,
            "username": username,
            "email": email,
            "telefono": telefono,
        }

        # Validaciones
        if not username or not email or not password or not password2:
            messages.error(request, "Completa todos los campos requeridos.")
            return render(request, "api/register.html", form_ctx)

        if password != password2:
            messages.error(request, "Las contraseñas no coinciden.")
            return render(request, "api/register.html", form_ctx)

        if len(password) < 6:
            messages.error(request, "La contraseña debe tener al menos 6 caracteres.")
            return render(request, "api/register.html", form_ctx)

        telefono, tel_error = validar_telefono(telefono)
        if tel_error:
            messages.error(request, tel_error)
            return render(request, "api/register.html", form_ctx)

        if Usuario.objects.filter(username=username).exists():
            messages.error(request, "El nombre de usuario ya está en uso.")
            form_ctx.pop("username", None)
            return render(request, "api/register.html", form_ctx)

        if Usuario.objects.filter(email=email).exists():
            messages.error(request, "El correo ya está registrado.")
            form_ctx.pop("email", None)
            return render(request, "api/register.html", form_ctx)

        # Crear usuario con contraseña hasheada
        usuario = Usuario(
            username=username,
            email=email,
            telefono=telefono or None,
            rol=rol,
            activo=True,
        )
        usuario.set_password(password)
        usuario.save()

        # Auto-login
        request.session['usuario_id'] = usuario.id_usuario
        request.session['usuario_rol'] = usuario.rol
        request.session['usuario_name'] = usuario.username
        request.session.set_expiry(0)

        messages.success(request, "Cuenta creada con éxito. ¡Bienvenido!")
        return redirect(next_url)

    return render(request, "api/register.html", {
        "next": request.GET.get('next', ''),
        "roles": allowed_roles,
        "selected_rol": "ANALISTA",
    })


# ---------------------------------------------------------------------------
# Vistas HTML — Gestión de Usuarios (solo ADMIN)
# ---------------------------------------------------------------------------

@usuario_login_required
@require_roles()
def usuarios_list_view(request):
    """Listado de todos los usuarios con estadísticas."""
    usuarios = Usuario.objects.all().order_by('-id_usuario')
    ctx = {
        "usuarios": usuarios,
        "activos": usuarios.filter(activo=True).count(),
        "inactivos": usuarios.filter(activo=False).count(),
        "admins": usuarios.filter(rol="ADMIN").count(),
    }
    return render(request, "api/usuarios.html", ctx)


@usuario_login_required
@require_roles()
def usuario_create_view(request):
    """Crear un nuevo usuario (admin)."""
    roles = [
        {"value": "ADMIN", "label": "Administrador"},
        {"value": "ANALISTA", "label": "Analista de Incidentes"},
        {"value": "GERENTE", "label": "Gerente (solo lectura)"},
    ]

    if request.method == "POST":
        username = (request.POST.get("username") or "").strip()
        email = (request.POST.get("email") or "").strip().lower()
        telefono = (request.POST.get("telefono") or "").strip()
        password = (request.POST.get("password") or "").strip()
        password2 = (request.POST.get("password2") or "").strip()
        rol = (request.POST.get("rol") or "ANALISTA").strip().upper()

        telefono, tel_error = validar_telefono(telefono)

        if not username or not email or not password:
            messages.error(request, "Completa todos los campos requeridos.")
        elif password != password2:
            messages.error(request, "Las contraseñas no coinciden.")
        elif len(password) < 6:
            messages.error(request, "La contraseña debe tener al menos 6 caracteres.")
        elif tel_error:
            messages.error(request, tel_error)
        elif Usuario.objects.filter(username=username).exists():
            messages.error(request, "Nombre de usuario ya en uso.")
        elif Usuario.objects.filter(email=email).exists():
            messages.error(request, "Email ya registrado. El sistema no permite continuar.")
        else:
            u = Usuario(username=username, email=email, telefono=telefono or None, rol=rol)
            u.set_password(password)
            u.save()
            messages.success(request, f"Usuario '{username}' creado correctamente.")
            return redirect("usuarios_list")

        return render(request, "api/usuario_form.html", {
            "editing": False,
            "roles": roles,
            "usuario": {"username": username, "email": email, "telefono": telefono, "rol": rol}
        })

    return render(request, "api/usuario_form.html", {
        "editing": False,
        "roles": roles,
    })


@usuario_login_required
@require_roles()
def usuario_edit_view(request, usuario_id):
    """Editar datos de un usuario (no cambia password)."""
    usuario = get_object_or_404(Usuario, id_usuario=usuario_id)
    roles = [
        {"value": "ADMIN", "label": "Administrador"},
        {"value": "ANALISTA", "label": "Analista de Incidentes"},
        {"value": "GERENTE", "label": "Gerente (solo lectura)"},
    ]

    if request.method == "POST":
        email = (request.POST.get("email") or "").strip().lower()
        telefono = (request.POST.get("telefono") or "").strip()
        rol = (request.POST.get("rol") or usuario.rol).strip().upper()
        activo = bool(request.POST.get("activo"))

        telefono, tel_error = validar_telefono(telefono)

        if not email:
            messages.error(request, "El email es requerido.")
        elif tel_error:
            messages.error(request, tel_error)
        elif Usuario.objects.filter(email=email).exclude(id_usuario=usuario_id).exists():
            messages.error(request, "Email ya registrado por otro usuario.")
        else:
            usuario.email = email
            usuario.telefono = telefono or None
            usuario.rol = rol
            usuario.activo = activo
            usuario.save()
            messages.success(request, f"Usuario '{usuario.username}' actualizado.")
            return redirect("usuarios_list")

    return render(request, "api/usuario_form.html", {
        "editing": True,
        "usuario": usuario,
        "roles": roles,
    })


@usuario_login_required
@require_roles()
def usuario_toggle_view(request, usuario_id):
    """Activar/desactivar un usuario (soft delete)."""
    if request.method == "POST":
        usuario = get_object_or_404(Usuario, id_usuario=usuario_id)
        # No permitir desactivar al propio admin
        if usuario.id_usuario == request.session.get("usuario_id"):
            messages.error(request, "No puedes desactivarte a ti mismo.")
        else:
            usuario.activo = not usuario.activo
            usuario.save(update_fields=["activo"])
            action_text = "activado" if usuario.activo else "desactivado"
            messages.success(request, f"Usuario '{usuario.username}' {action_text}.")
    return redirect("usuarios_list")


@usuario_login_required
@require_roles()
def usuario_reset_password_view(request, usuario_id):
    """Resetear la contraseña de un usuario."""
    if request.method == "POST":
        usuario = get_object_or_404(Usuario, id_usuario=usuario_id)
        new_password = (request.POST.get("new_password") or "").strip()

        if len(new_password) < 6:
            messages.error(request, "La contraseña debe tener al menos 6 caracteres.")
        else:
            usuario.set_password(new_password)
            usuario.save(update_fields=["password"])
            messages.success(
                request,
                f"Contraseña de '{usuario.username}' reseteada correctamente.",
            )
    return redirect("usuarios_list")


# ---------------------------------------------------------------------------
# Vistas HTML — Exportación PDF
# ---------------------------------------------------------------------------

@usuario_login_required
@require_roles('GERENTE')
def dashboard_pdf_view(request):
    """Exporta el dashboard como PDF."""
    pendientes = Incidente.objects.filter(estado="Pendiente").count()
    confirmados = Incidente.objects.filter(estado="Fraude confirmado").count()
    falsos = Incidente.objects.filter(estado="Falso positivo").count()
    recientes = Incidente.objects.order_by("-fecha")[:20]

    return build_dashboard_pdf({
        "pendientes": pendientes,
        "confirmados": confirmados,
        "falsos": falsos,
        "recientes": recientes,
    })


@usuario_login_required
@require_roles('ANALISTA')
def incidentes_pdf_view(request):
    """Exporta la lista de incidentes como PDF."""
    incidentes = Incidente.objects.all().order_by('-fecha')
    return build_incidentes_pdf(incidentes)


@usuario_login_required
@require_roles('ANALISTA')
def incidente_detalle_pdf_view(request, incidente_id):
    """Exporta un incidente individual como PDF."""
    incidente = get_object_or_404(Incidente, id_incidente=incidente_id)
    return build_incidente_detalle_pdf(incidente)

