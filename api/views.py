from django.shortcuts import render

from rest_framework import viewsets
from .models import Usuario
from .serializers import UsuarioSerializer
from .models import Transaccion
from .serializers import TransaccionSerializer
class UsuarioViewSet(viewsets.ModelViewSet):
    queryset = Usuario.objects.all()
    serializer_class = UsuarioSerializer

class TransaccionViewSet(viewsets.ModelViewSet):
    queryset = Transaccion.objects.all()
    serializer_class = TransaccionSerializer
