import csv, io
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.csrf import csrf_exempt
from django_filters.rest_framework import DjangoFilterBackend
from django.contrib import messages
from rest_framework import filters, status, viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import action
from .ml_utils import predict_fraud
from .serializers import UsuarioSerializer, TransaccionSerializer, IncidenteSerializer, ConfiguracionSerializer
from .models import Transaccion, Incidente, Usuario, Configuracion
import json
from django.db.models.functions import TruncDate
from django.db.models import Count
from django.utils import timezone
from functools import wraps
from django.urls import reverse

ROLE_LABELS = {
    "ADMIN": "Administrador",
    "ANALISTA": "Analista de Incidentes",
    "EJECUTIVO": "Ejecutivo (solo lectura)",
}
def _allowed_roles_for_request(request):
    is_admin = (request.session.get('usuario_rol') == 'ADMIN')
    roles = ["ADMIN", "ANALISTA", "EJECUTIVO"] if is_admin else ["ANALISTA", "EJECUTIVO"]
    return [{"value": r, "label": ROLE_LABELS.get(r, r)} for r in roles]


def usuario_login_required(view_func):
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if not request.session.get('usuario_id'):
            next_url = request.get_full_path()
            return redirect(f"{reverse('login')}?next={next_url}")
        return view_func(request, *args, **kwargs)

    return _wrapped
def require_roles(*allowed_roles):
    """
    Restringe acceso por rol. ADMIN siempre pasa.
    Uso: @usuario_login_required @require_roles('ANALISTA','EJECUTIVO')
    """
    def deco(view_func):
        @wraps(view_func)
        def _wrapped(request, *args, **kwargs):
            rol = request.session.get('usuario_rol')
            if not rol:
                return redirect(f"{reverse('login')}?next={request.get_full_path()}")
            if rol == 'ADMIN' or rol in allowed_roles:
                return view_func(request, *args, **kwargs)
            messages.error(request, "No tienes permisos para acceder a este módulo.")
            return redirect('inicio')
        return _wrapped
    return deco

class UsuarioViewSet(viewsets.ModelViewSet):
    queryset = Usuario.objects.all()
    serializer_class = UsuarioSerializer


class TransaccionViewSet(viewsets.ModelViewSet):
    queryset = Transaccion.objects.all()
    serializer_class = TransaccionSerializer
    filter_backends = [DjangoFilterBackend]
    # ⬇️ ahora filtramos por los campos del modelo
    filterset_fields = ['category', 'state', 'gender']

    def create(self, request, *args, **kwargs):
        # Guardamos la transacción
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        transaccion = serializer.save()

        # Ejecutar predicción con los campos del modelo
        # (ml_utils.predict_fraud deriva amt_log1p, hour, weekday, month, is_weekend)
        score, explicabilidad = predict_fraud(transaccion)

        # Umbral dinámico
        config, _ = Configuracion.objects.get_or_create(id=1, defaults={"umbral_score": 70})
        umbral = config.umbral_score

        # Crear incidente si supera el umbral
        if score >= umbral:
            Incidente.objects.create(
                id_transaccion=transaccion,
                score_riesgo=score,
                explicabilidad=explicabilidad,
                estado="Pendiente"
            )

        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


class IncidenteViewSet(viewsets.ModelViewSet):
    queryset = Incidente.objects.all().order_by('-fecha')
    serializer_class = IncidenteSerializer
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['estado']
    ordering_fields = ['score_riesgo', 'fecha']  # 👈 permitimos ordenar
    ordering = ['-fecha']

    @action(detail=True, methods=['patch'])
    def cambiar_estado(self, request, pk=None):
        incidente = self.get_object()
        nuevo_estado = request.data.get("estado")

        if nuevo_estado not in ["Fraude confirmado", "Falso positivo"]:
            return Response(
                {"error": "Estado no válido. Usa 'Fraude confirmado' o 'Falso positivo'."},
                status=status.HTTP_400_BAD_REQUEST
            )

        incidente.estado = nuevo_estado
        incidente.comentario = request.data.get("comentario", "")
        incidente.save()

        return Response(
            {"mensaje": f"Incidente {incidente.id_incidente} actualizado a {incidente.estado}."},
            status=status.HTTP_200_OK
        )

    @action(detail=False, methods=['get'])
    def since(self, request):
        """
        Devuelve incidentes con id_incidente > after_id (máx 50),
        ordenados desc por fecha. Ideal para polling ligero.
        """
        try:
            after = int(request.query_params.get('after_id', 0))
        except (TypeError, ValueError):
            after = 0

        qs = Incidente.objects.filter(id_incidente__gt=after) \
            .order_by('-fecha')[:50]

        items = [{
            "id_incidente": i.id_incidente,
            "score_riesgo": float(i.score_riesgo or 0),
            "estado": i.estado,
            "fecha": i.fecha.isoformat(),
        } for i in qs]

        return Response({"items": items})


class AuditoriaView(APIView):
    def post(self, request):
        data = request.data

        # Si es un solo objeto, lo procesamos como lista de 1
        if isinstance(data, dict):
            data = [data]

        resultados = []
        for transaccion in data:
            score, explicabilidad = predict_fraud(transaccion)
            resultados.append({
                "transaccion": transaccion,
                "score_riesgo": score,
                "explicabilidad": explicabilidad,
                "mensaje": "Predicción temporal, no guardada en BD."
            })

        return Response(resultados, status=status.HTTP_200_OK)


class ConfiguracionViewSet(viewsets.ModelViewSet):
    queryset = Configuracion.objects.all()
    serializer_class = ConfiguracionSerializer


@usuario_login_required
@require_roles('EJECUTIVO')
def dashboard_view(request):
    # KPIs
    pendientes = Incidente.objects.filter(estado="Pendiente").count()
    confirmados = Incidente.objects.filter(estado="Fraude confirmado").count()
    falsos = Incidente.objects.filter(estado="Falso positivo").count()

    recientes = Incidente.objects.order_by("-fecha")[:10]

    # ---------- Fraude por ciudad (solo confirmados)
    qs_city = (Incidente.objects
               .filter(estado="Fraude confirmado")
               .values("id_transaccion__state")
               .annotate(n=Count("id_incidente"))
               .order_by("-n"))
    labels_city, data_city, otros = [], [], 0
    for i, r in enumerate(qs_city):
        name = r["id_transaccion__state"] or "—"
        if i < 10:
            labels_city.append(name)
            data_city.append(r["n"])
        else:
            otros += r["n"]
    if otros:
        labels_city.append("Otros")
        data_city.append(otros)

    # ---------- Fraude por categoría (solo confirmados)
    qs_cat = (Incidente.objects
              .filter(estado="Fraude confirmado")
              .values("id_transaccion__category")
              .annotate(n=Count("id_incidente"))
              .order_by("-n"))
    labels_cat = [(r["id_transaccion__category"] or "—") for r in qs_cat]
    data_cat = [r["n"] for r in qs_cat]

    # ---------- Distribución de score (buckets)
    buckets = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    bucket_labels = ["0–20", "20–40", "40–60", "60–80", "80–100"]
    bucket_counts = []
    for lo, hi in buckets:
        if hi < 100:
            c = Incidente.objects.filter(score_riesgo__gte=lo, score_riesgo__lt=hi).count()
        else:
            c = Incidente.objects.filter(score_riesgo__gte=lo, score_riesgo__lte=hi).count()
        bucket_counts.append(c)

    # ---------- Tendencia últimos 30 días (por estado)
    tz = timezone.get_current_timezone()
    since = timezone.now() - timezone.timedelta(days=29)
    daily = (Incidente.objects.filter(fecha__date__gte=since.date())
             .annotate(d=TruncDate("fecha", tzinfo=tz))
             .values("d", "estado")
             .annotate(n=Count("id_incidente"))
             .order_by("d"))

    dates = sorted({row["d"] for row in daily})
    date_labels = [d.strftime("%d/%b") for d in dates]
    estados = ["Pendiente", "Fraude confirmado", "Falso positivo"]
    series = {e: [0] * len(dates) for e in estados}
    index = {d: i for i, d in enumerate(dates)}
    for row in daily:
        i = index[row["d"]]
        series[row["estado"]][i] = row["n"]

    ctx = {
        "pendientes": pendientes,
        "confirmados": confirmados,
        "falsos": falsos,
        "recientes": recientes,

        "labels_city": json.dumps(labels_city),
        "data_city": json.dumps(data_city),

        "labels_cat": json.dumps(labels_cat),
        "data_cat": json.dumps(data_cat),

        "bucket_labels": json.dumps(bucket_labels),
        "bucket_counts": json.dumps(bucket_counts),

        "date_labels": json.dumps(date_labels),
        "serie_pendiente": json.dumps(series["Pendiente"]),
        "serie_fraude": json.dumps(series["Fraude confirmado"]),
        "serie_fp": json.dumps(series["Falso positivo"]),
    }
    return render(request, "api/dashboard.html", ctx)


@usuario_login_required
@require_roles('ANALISTA')
def incidentes_view(request):
    incidentes = Incidente.objects.all().order_by('-fecha')
    return render(request, "api/incidentes.html", {"incidentes": incidentes})


@usuario_login_required
@require_roles('ANALISTA')
def incidente_detalle_view(request, incidente_id):
    incidente = get_object_or_404(Incidente, id_incidente=incidente_id)

    if request.method == "POST":
        nuevo_estado = request.POST.get("estado")
        comentario = request.POST.get("comentario", "")

        if nuevo_estado in ["Fraude confirmado", "Falso positivo"]:
            incidente.estado = nuevo_estado
            incidente.comentario = comentario
            incidente.save()
            messages.success(request,
                             f"Incidente #{incidente.id_incidente} actualizado correctamente a {nuevo_estado}.")
            return redirect("incidentes_listado")

    return render(request, "api/incidente_detalle.html", {"incidente": incidente})


@usuario_login_required
@csrf_exempt
@require_roles('ANALISTA','EJECUTIVO')
def auditoria_view(request):
    contexto = {"active_tab": "individual"}
    if request.method == "POST":
        try:
            importe = float(request.POST.get("importe") or 0)
        except ValueError:
            messages.error(request, "Importe inválido.")
            return render(request, "api/auditoria.html", contexto)

        payload = {
            "importe": importe,
            "category": (request.POST.get("category") or "").strip(),
            "state": (request.POST.get("state") or "").strip(),
            "gender": (request.POST.get("gender") or "").strip().lower(),
            "age": int(request.POST.get("age") or 0),
            "city_pop": int(request.POST.get("city_pop") or 0),
            # 'fecha' no es necesario: se deriva 'hour/weekday/month/is_weekend' internamente si no viene
        }

        score, explicabilidad = predict_fraud(payload)
        contexto.update({
            "resultado": {"score_riesgo": score, "explicabilidad": explicabilidad},
            "active_tab": "individual"
        })
        messages.success(request, "Auditoría individual procesada.")
    return render(request, "api/auditoria.html", contexto)


@usuario_login_required
@require_roles('ANALISTA','EJECUTIVO')
def auditoria_lote_view(request):
    contexto = {"active_tab": "lote"}
    if request.method == "POST":
        archivo = request.FILES.get("archivo")
        if not archivo:
            messages.error(request, "Debes subir un archivo CSV válido.")
            return render(request, "api/auditoria.html", contexto)

        try:
            data = archivo.read().decode("utf-8", errors="ignore")
            f = io.StringIO(data)
            reader = csv.DictReader(f)

            expected = {"importe", "category", "state", "gender", "age", "city_pop"}
            headers = set([h.strip() for h in (reader.fieldnames or [])])
            if not expected.issubset(headers):
                messages.error(request, "Cabeceras inválidas. Se esperan: importe,category,state,gender,age,city_pop")
                return render(request, "api/auditoria.html", contexto)

            resultados = []
            for row in reader:
                try:
                    importe = float(str(row.get("importe", "")).replace(",", "."))
                    age = int(row.get("age") or 0)
                    city_pop = int(row.get("city_pop") or 0)
                except ValueError:
                    continue

                payload = {
                    "importe": importe,
                    "category": (row.get("category") or "").strip(),
                    "state": (row.get("state") or "").strip(),
                    "gender": (row.get("gender") or "").strip().lower(),
                    "age": age,
                    "city_pop": city_pop,
                }

                score, explicabilidad = predict_fraud(payload)
                resultados.append({
                    **payload,
                    "score_riesgo": score,
                    "explicabilidad": explicabilidad,
                })

            resultados.sort(key=lambda x: x["score_riesgo"], reverse=True)
            contexto.update({"resultados": resultados, "active_tab": "lote"})
            messages.success(request, f"Auditoría por lote procesada. Filas válidas: {len(resultados)}")
        except Exception as e:
            messages.error(request, f"Error procesando el CSV: {e}")

    return render(request, "api/auditoria.html", contexto)


@usuario_login_required
@require_roles('EJECUTIVO')
def configuracion_front(request):
    config, _ = Configuracion.objects.get_or_create(id=1)

    if request.method == "POST":
        nuevo_umbral = int(request.POST.get("umbral_score"))
        id_usuario = request.POST.get("actualizado_por")

        config.umbral_score = nuevo_umbral
        if id_usuario:
            try:
                usuario = Usuario.objects.get(id_usuario=id_usuario)
                config.actualizado_por = usuario
            except Usuario.DoesNotExist:
                pass
        config.save()
        messages.success(request, "Configuración actualizada correctamente.")
        return redirect("configuracion_front")

    return render(
        request,
        "api/configuracion.html",
        {"config": config, "usuarios": Usuario.objects.all()}
    )


def inicio_view(request):
    pendientes = Incidente.objects.filter(estado="Pendiente").count()
    confirmados = Incidente.objects.filter(estado="Fraude confirmado").count()
    falsos = Incidente.objects.filter(estado="Falso positivo").count()

    contexto = {
        "pendientes": pendientes,
        "confirmados": confirmados,
        "falsos": falsos,
    }
    return render(request, "api/inicio.html", contexto)


def ayuda_view(request):
    config, _ = Configuracion.objects.get_or_create(id=1)
    return render(request, "api/ayuda.html", {"config": config})


def login_view(request):
    # Si ya está logueado, redirigimos
    if request.session.get('usuario_id'):
        return redirect(request.GET.get('next') or reverse('inicio'))

    if request.method == "POST":
        username_or_email = (request.POST.get('username') or '').strip()
        password = (request.POST.get('password') or '').strip()
        remember = bool(request.POST.get('remember'))
        next_url = request.POST.get('next') or request.GET.get('next') or reverse('inicio')

        from .models import Usuario
        usuario = None
        # Permite loguear con username o email
        for field in ('username', 'email'):
            try:
                usuario = Usuario.objects.get(**{field: username_or_email}, activo=True)
                break
            except Usuario.DoesNotExist:
                pass

        if not usuario:
            messages.error(request, "Usuario no encontrado o inactivo.")
            return render(request, "api/login.html", {"next": next_url})

        # ⚠️ Por ahora compara texto plano (tú guardas 'password' en claro).
        #     Cuando quieras pasamos a hashing (PBKDF2/BCrypt).
        if usuario.password != password:
            messages.error(request, "Contraseña incorrecta.")
            return render(request, "api/login.html", {"next": next_url})

        # Sesión
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
    request.session.flush()
    messages.success(request, "Sesión cerrada.")
    return redirect('login')

def register_view(request):
    # Si ya está logueado, redirigimos
    if request.session.get('usuario_id'):
        return redirect(request.GET.get('next') or reverse('inicio'))

    allowed_roles = _allowed_roles_for_request(request)

    if request.method == "POST":
        username = (request.POST.get('username') or '').strip()
        email = (request.POST.get('email') or '').strip().lower()
        telefono = (request.POST.get('telefono') or '').strip()
        password = (request.POST.get('password') or '').strip()
        password2 = (request.POST.get('password2') or '').strip()
        rol = ((request.POST.get('rol') or 'ANALISTA').strip().upper())
        next_url = request.POST.get('next') or request.GET.get('next') or reverse('inicio')

        # Validar rol permitido por contexto
        allowed_values = {r["value"] for r in allowed_roles}
        if rol not in allowed_values:
            rol = "ANALISTA"  # fallback seguro

        # Validaciones mínimas
        if not username or not email or not password or not password2:
            messages.error(request, "Completa todos los campos requeridos.")
            return render(request, "api/register.html", {
                "next": next_url, "roles": allowed_roles, "selected_rol": rol,
                "username": username, "email": email, "telefono": telefono
            })
        if password != password2:
            messages.error(request, "Las contraseñas no coinciden.")
            return render(request, "api/register.html", {
                "next": next_url, "roles": allowed_roles, "selected_rol": rol,
                "username": username, "email": email, "telefono": telefono
            })
        if len(password) < 6:
            messages.error(request, "La contraseña debe tener al menos 6 caracteres.")
            return render(request, "api/register.html", {
                "next": next_url, "roles": allowed_roles, "selected_rol": rol,
                "username": username, "email": email, "telefono": telefono
            })
        if Usuario.objects.filter(username=username).exists():
            messages.error(request, "El nombre de usuario ya está en uso.")
            return render(request, "api/register.html", {
                "next": next_url, "roles": allowed_roles, "selected_rol": rol,
                "email": email, "telefono": telefono
            })
        if Usuario.objects.filter(email=email).exists():
            messages.error(request, "El correo ya está registrado.")
            return render(request, "api/register.html", {
                "next": next_url, "roles": allowed_roles, "selected_rol": rol,
                "username": username, "telefono": telefono
            })

        # ⚠️ Hoy guardas password en texto plano; cuando quieras migramos a hash.
        usuario = Usuario.objects.create(
            username=username,
            email=email,
            telefono=telefono or None,
            password=password,
            rol=rol,
            activo=True
        )

        # Auto-login
        request.session['usuario_id'] = usuario.id_usuario
        request.session['usuario_rol'] = usuario.rol
        request.session['usuario_name'] = usuario.username
        request.session.set_expiry(0)

        messages.success(request, "Cuenta creada con éxito. ¡Bienvenido!")
        return redirect(next_url)

    # GET
    return render(request, "api/register.html", {
        "next": request.GET.get('next', ''),
        "roles": allowed_roles,
        "selected_rol": "ANALISTA",
    })

