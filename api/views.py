import csv, io
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.csrf import csrf_exempt
from django_filters.rest_framework import DjangoFilterBackend
from django.contrib import messages
from rest_framework import filters,status, viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import action
from .ml_utils import predict_fraud
from .serializers import UsuarioSerializer, TransaccionSerializer, IncidenteSerializer, ConfiguracionSerializer
from .models import Transaccion, Incidente, Usuario, Configuracion


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
def dashboard_view(request):
    # Contar incidentes por estado
    pendientes = Incidente.objects.filter(estado="Pendiente").count()
    confirmados = Incidente.objects.filter(estado="Fraude confirmado").count()
    falsos = Incidente.objects.filter(estado="Falso positivo").count()

    # Últimos 5 incidentes
    recientes = Incidente.objects.all().order_by('-fecha')[:5]

    contexto = {
        "pendientes": pendientes,
        "confirmados": confirmados,
        "falsos": falsos,
        "recientes": recientes,
    }
    return render(request, "api/dashboard.html", contexto)

def incidentes_view(request):
    incidentes = Incidente.objects.all().order_by('-fecha')
    return render(request, "api/incidentes.html", {"incidentes": incidentes})

def incidente_detalle_view(request, incidente_id):
    incidente = get_object_or_404(Incidente, id_incidente=incidente_id)

    if request.method == "POST":
        nuevo_estado = request.POST.get("estado")
        comentario = request.POST.get("comentario", "")

        if nuevo_estado in ["Fraude confirmado", "Falso positivo"]:
            incidente.estado = nuevo_estado
            incidente.comentario = comentario
            incidente.save()
            messages.success(request, f"Incidente #{incidente.id_incidente} actualizado correctamente a {nuevo_estado}.")
            return redirect("incidentes_listado")

    return render(request, "api/incidente_detalle.html", {"incidente": incidente})
@csrf_exempt
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
            "state":    (request.POST.get("state") or "").strip(),
            "gender":   (request.POST.get("gender") or "").strip().lower(),
            "age":      int(request.POST.get("age") or 0),
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

            expected = {"importe","category","state","gender","age","city_pop"}
            headers = set([h.strip() for h in (reader.fieldnames or [])])
            if not expected.issubset(headers):
                messages.error(request, "Cabeceras inválidas. Se esperan: importe,category,state,gender,age,city_pop")
                return render(request, "api/auditoria.html", contexto)

            resultados = []
            for row in reader:
                try:
                    importe = float(str(row.get("importe","")).replace(",", "."))
                    age = int(row.get("age") or 0)
                    city_pop = int(row.get("city_pop") or 0)
                except ValueError:
                    continue

                payload = {
                    "importe": importe,
                    "category": (row.get("category") or "").strip(),
                    "state":    (row.get("state") or "").strip(),
                    "gender":   (row.get("gender") or "").strip().lower(),
                    "age":      age,
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
