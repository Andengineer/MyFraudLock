from django.contrib import admin
from .models import Usuario, Transaccion, Incidente, Configuracion


@admin.register(Usuario)
class UsuarioAdmin(admin.ModelAdmin):
    list_display = ('id_usuario', 'username', 'email', 'rol', 'activo')
    list_filter = ('rol', 'activo')
    search_fields = ('username', 'email')
    exclude = ('password',)  # nunca mostrar/editar password desde admin


@admin.register(Transaccion)
class TransaccionAdmin(admin.ModelAdmin):
    list_display = ('id_transaccion', 'importe', 'card_brand', 'customer_region', 'category', 'fecha')
    list_filter = ('card_brand', 'customer_region', 'card_type')
    search_fields = ('category', 'customer_region', 'issuer_bank')
    date_hierarchy = 'fecha'


@admin.register(Incidente)
class IncidenteAdmin(admin.ModelAdmin):
    list_display = ('id_incidente', 'estado', 'score_riesgo', 'fecha', 'id_transaccion')
    list_filter = ('estado',)
    search_fields = ('id_incidente', 'comentario')
    date_hierarchy = 'fecha'


@admin.register(Configuracion)
class ConfiguracionAdmin(admin.ModelAdmin):
    list_display = ('id', 'umbral_score', 'actualizado_en', 'actualizado_por')
