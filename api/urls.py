from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UsuarioViewSet, TransaccionViewSet, IncidenteViewSet

router = DefaultRouter()
router.register(r'usuarios', UsuarioViewSet)
router.register(r'transacciones', TransaccionViewSet)
router.register(r'incidentes', IncidenteViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
