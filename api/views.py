from django.shortcuts import render
from .serializers import UsuarioSerializer, TransaccionSerializer, IncidenteSerializer
from .models import Transaccion, Incidente, Usuario
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters,status, viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from .ml_utils import predict_fraud

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

class AuditoriaView(APIView):
    def post(self, request):
        # Ejecutar predicción sin guardar en BD
        score, explicabilidad = predict_fraud(request.data)

        return Response({
            "score_riesgo": score,
            "explicabilidad": explicabilidad,
            "mensaje": "Esto es solo una predicción temporal, no se guardó en la base de datos."
        }, status=status.HTTP_200_OK)


