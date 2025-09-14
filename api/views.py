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
    filterset_fields = ['metodo_pago']

    def create(self, request, *args, **kwargs):
        # Guardamos la transacción
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        transaccion = serializer.save()

        # Ejecutar predicción
        score, explicabilidad = predict_fraud(serializer.data)

        # Obtener umbral dinámico (si no existe, se crea con 70)
        config, _ = Configuracion.objects.get_or_create(id=1, defaults={"umbral_score": 70})
        umbral = config.umbral_score

        # Evaluar score contra umbral
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
    resultado = None

    if request.method == "POST":
        importe = request.POST.get("importe")
        metodo_pago = request.POST.get("metodo_pago")
        direccion_envio = request.POST.get("direccion_envio")

        # Ejecutar predicción simulada
        score, explicabilidad = predict_fraud({
            "importe": importe,
            "metodo_pago": metodo_pago,
            "direccion_envio": direccion_envio
        })

        resultado = {
            "score_riesgo": score,
            "explicabilidad": explicabilidad,
        }

    return render(request, "api/auditoria.html", {"resultado": resultado})

def auditoria_lote_view(request):
    resultados = []

    if request.method == "POST" and request.FILES.get("archivo"):
        archivo = request.FILES["archivo"]

        # Validación básica
        if not archivo.name.lower().endswith(".csv"):
            messages.error(request, "Por favor sube un archivo CSV.")
            return render(request, "api/auditoria_lote.html", {"resultados": resultados})

        try:
            # Decodificar a texto y leer CSV
            wrapper = io.TextIOWrapper(archivo.file, encoding="utf-8")
            reader = csv.DictReader(wrapper)

            # Validar columnas requeridas
            columnas_req = {"importe", "metodo_pago", "direccion_envio"}
            if not columnas_req.issubset(set([c.strip() for c in reader.fieldnames or []])):
                messages.error(
                    request,
                    "El CSV debe tener las columnas: importe, metodo_pago, direccion_envio."
                )
                return render(request, "api/auditoria_lote.html", {"resultados": resultados})

            # Procesar filas
            for row in reader:
                datos = {
                    "importe": row.get("importe"),
                    "metodo_pago": row.get("metodo_pago"),
                    "direccion_envio": row.get("direccion_envio"),
                }
                score, explicabilidad = predict_fraud(datos)
                resultados.append({
                    "importe": datos["importe"],
                    "metodo_pago": datos["metodo_pago"],
                    "direccion_envio": datos["direccion_envio"],
                    "score_riesgo": score,
                    "explicabilidad": explicabilidad,
                })

            # Ordenar de mayor a menor score para priorizar revisión
            resultados.sort(key=lambda x: x["score_riesgo"], reverse=True)
            messages.success(request, f"Se procesaron {len(resultados)} transacciones (no se guardó en BD).")

        except Exception as e:
            messages.error(request, f"Error al procesar el CSV: {e}")

    return render(request, "api/auditoria_lote.html", {"resultados": resultados})

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
