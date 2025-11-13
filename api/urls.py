from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UsuarioViewSet, TransaccionViewSet, IncidenteViewSet, AuditoriaView, dashboard_view, incidentes_view, \
    incidente_detalle_view, auditoria_view, auditoria_lote_view, configuracion_front, ConfiguracionViewSet, inicio_view, \
    ayuda_view, login_view, logout_view, register_view

router = DefaultRouter()
router.register(r'usuarios', UsuarioViewSet)
router.register(r'transacciones', TransaccionViewSet)
router.register(r'incidentes', IncidenteViewSet)
router.register(r'configuracion', ConfiguracionViewSet)

urlpatterns = [

    # ==== FRONTEND PRINCIPAL (HTML) ====
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('register/', register_view, name='register'),
    path('inicio/', inicio_view, name='inicio'),
    path('dashboard/', dashboard_view, name='dashboard'),
    path('incidentes-listado/', incidentes_view, name='incidentes_listado'),
    path('incidentes/<int:incidente_id>/detalle/', incidente_detalle_view, name='incidente_detalle'),
    path('auditoria/', auditoria_view, name='auditoria_front'),
    path('auditoria-api/', AuditoriaView.as_view(), name='auditoria_api'),
    path('auditoria-lote/', auditoria_lote_view, name='auditoria_lote'),
    path('configuracion-front/', configuracion_front, name='configuracion_front'),
    path('ayuda/', ayuda_view, name='ayuda'),

    # ==== API REST ====
    path('', include(router.urls)),
]