from django.shortcuts import render

from rest_framework import viewsets
from .models import Usuario
from .serializers import UsuarioSerializer
from .models import Transaccion
from .serializers import TransaccionSerializer
from .models import Incidente
from .serializers import IncidenteSerializer
class UsuarioViewSet(viewsets.ModelViewSet):
    queryset = Usuario.objects.all()
    serializer_class = UsuarioSerializer

class TransaccionViewSet(viewsets.ModelViewSet):
    queryset = Transaccion.objects.all()
    serializer_class = TransaccionSerializer

class IncidenteViewSet(viewsets.ModelViewSet):
    queryset = Incidente.objects.all()
    serializer_class = IncidenteSerializer