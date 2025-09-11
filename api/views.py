from django.shortcuts import render

from rest_framework import viewsets
from .models import Usuario
from .serializers import UsuarioSerializer
from .models import Transaccion
from .serializers import TransaccionSerializer
from .models import Incidente
from .serializers import IncidenteSerializer
from rest_framework import filters
from django_filters.rest_framework import DjangoFilterBackend
class UsuarioViewSet(viewsets.ModelViewSet):
    queryset = Usuario.objects.all()
    serializer_class = UsuarioSerializer

class TransaccionViewSet(viewsets.ModelViewSet):
    queryset = Transaccion.objects.all()
    serializer_class = TransaccionSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['metodo_pago']

class IncidenteViewSet(viewsets.ModelViewSet):
    queryset = Incidente.objects.all().order_by('-fecha')
    serializer_class = IncidenteSerializer
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['estado']
    ordering_fields = ['score_riesgo', 'fecha']  # 👈 permitimos ordenar
    ordering = ['-fecha']

