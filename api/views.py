from django.shortcuts import render, get_object_or_404, redirect
from django_filters.rest_framework import DjangoFilterBackend
from django.contrib import messages
from rest_framework import filters,status, viewsets
from rest_framework.decorators import action
from rest_framework.views import APIView
from rest_framework.response import Response
from .ml_utils import predict_fraud
from .serializers import UsuarioSerializer, TransaccionSerializer, IncidenteSerializer
from .models import Transaccion, Incidente, Usuario

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

        # Umbral de riesgo
        if score >= 70:
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