from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UsuarioViewSet, TransaccionViewSet, IncidenteViewSet, AuditoriaView, dashboard_view, incidentes_view, incidente_detalle_view, auditoria_view, auditoria_lote_view, configuracion_view, ConfiguracionViewSet

router = DefaultRouter()
router.register(r'usuarios', UsuarioViewSet)
router.register(r'transacciones', TransaccionViewSet)
router.register(r'incidentes', IncidenteViewSet)
router.register(r'configuracion', ConfiguracionViewSet)


urlpatterns = [
    path('', include(router.urls)),
    path('auditoria-api/', AuditoriaView.as_view(), name='auditoria_api'),  # API JSON
    path('auditoria/', auditoria_view, name='auditoria_front'),  # Frontend HTML
    path('auditoria-lote/', auditoria_lote_view, name='auditoria_lote'),  # 👈 nuevo
    path('dashboard/', dashboard_view, name='dashboard'),
    path('incidentes-listado/', incidentes_view, name='incidentes_listado'),
    path('incidentes/<int:incidente_id>/detalle/', incidente_detalle_view, name='incidente_detalle'),
    path('configuracion/', configuracion_view, name='configuracion'),
]
