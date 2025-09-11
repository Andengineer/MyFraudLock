from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UsuarioViewSet, TransaccionViewSet, IncidenteViewSet, AuditoriaView, dashboard_view, incidentes_view, incidente_detalle_view

router = DefaultRouter()
router.register(r'usuarios', UsuarioViewSet)
router.register(r'transacciones', TransaccionViewSet)
router.register(r'incidentes', IncidenteViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('auditoria/', AuditoriaView.as_view(), name='auditoria'),  # 👈 nuevo endpoint
    path('dashboard/', dashboard_view, name='dashboard'),
    path('incidentes-listado/', incidentes_view, name='incidentes_listado'),
    path('incidentes/<int:incidente_id>/detalle/', incidente_detalle_view, name='incidente_detalle'),
]
