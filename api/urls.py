from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    # REST ViewSets
    UsuarioViewSet, TransaccionViewSet, IncidenteViewSet, ConfiguracionViewSet,
    SimulacionView,
    # HTML views — Auth
    login_view, logout_view, register_view,
    # HTML views — Pages
    inicio_view, ayuda_view,
    dashboard_view, incidentes_view, incidente_detalle_view,
    simulacion_view, simulacion_lote_view, configuracion_front,
    # HTML views — User management (ADMIN)
    usuarios_list_view, usuario_create_view, usuario_edit_view,
    usuario_toggle_view, usuario_reset_password_view,
    # HTML views — PDF export
    dashboard_pdf_view, incidentes_pdf_view, incidente_detalle_pdf_view,
)

router = DefaultRouter()
router.register(r'usuarios', UsuarioViewSet)
router.register(r'transacciones', TransaccionViewSet)
router.register(r'incidentes', IncidenteViewSet)
router.register(r'configuracion', ConfiguracionViewSet)

urlpatterns = [
    # ==== AUTH ====
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('register/', register_view, name='register'),

    # ==== FRONTEND PRINCIPAL (HTML) ====
    path('inicio/', inicio_view, name='inicio'),
    path('dashboard/', dashboard_view, name='dashboard'),
    path('incidentes-listado/', incidentes_view, name='incidentes_listado'),
    path('incidentes/<int:incidente_id>/detalle/', incidente_detalle_view, name='incidente_detalle'),
    path('simulacion/', simulacion_view, name='simulacion_front'),
    path('simulacion-api/', SimulacionView.as_view(), name='simulacion_api'),
    path('simulacion-lote/', simulacion_lote_view, name='simulacion_lote'),
    path('configuracion-front/', configuracion_front, name='configuracion_front'),
    path('ayuda/', ayuda_view, name='ayuda'),

    # ==== GESTIÓN DE USUARIOS (ADMIN) ====
    path('usuarios-panel/', usuarios_list_view, name='usuarios_list'),
    path('usuarios-panel/crear/', usuario_create_view, name='usuario_create'),
    path('usuarios-panel/<int:usuario_id>/editar/', usuario_edit_view, name='usuario_edit'),
    path('usuarios-panel/<int:usuario_id>/toggle/', usuario_toggle_view, name='usuario_toggle'),
    path('usuarios-panel/<int:usuario_id>/reset-password/', usuario_reset_password_view, name='usuario_reset_password'),

    # ==== EXPORTACIÓN PDF ====
    path('dashboard/pdf/', dashboard_pdf_view, name='dashboard_pdf'),
    path('incidentes-listado/pdf/', incidentes_pdf_view, name='incidentes_pdf'),
    path('incidentes/<int:incidente_id>/pdf/', incidente_detalle_pdf_view, name='incidente_detalle_pdf'),

    # ==== API REST ====
    path('', include(router.urls)),
]