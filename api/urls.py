from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UsuarioViewSet
from .views import UsuarioViewSet, TransaccionViewSet

router = DefaultRouter()
router.register(r'usuarios', UsuarioViewSet)
router.register(r'transacciones', TransaccionViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
