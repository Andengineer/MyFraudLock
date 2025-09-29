from django.db import models
from django.contrib.auth.models import User


class Usuario(models.Model):
    class Roles(models.TextChoices):
        ADMIN = 'ADMIN', 'Administrador'
        ANALISTA = 'ANALISTA', 'Analista de Fraude'
        EJECUTIVO = 'EJECUTIVO', 'Ejecutivo (solo lectura)'

    id_usuario = models.AutoField(primary_key=True)
    username = models.CharField(max_length=100, null=False)
    email = models.EmailField(unique=True, null=False)
    password = models.CharField(max_length=100, null=False)
    telefono = models.CharField(max_length=9, null=True, blank=True)
    activo = models.BooleanField(default=True)
    rol = models.CharField(
        max_length=15,
        choices=Roles.choices,
        default=Roles.ANALISTA,
        db_index=True,
    )

    def __str__(self):
        return self.username


class Transaccion(models.Model):
    id_transaccion = models.AutoField(primary_key=True)
    importe = models.DecimalField(max_digits=10, decimal_places=3, null=False)
    metodo_pago = models.CharField(max_length=50, null=False)
    direccion_envio = models.CharField(max_length=100, null=False)
    fecha = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Transacción {self.id_transaccion} - {self.metodo_pago} - {self.importe}"


class Incidente(models.Model):
    id_incidente = models.AutoField(primary_key=True)
    id_usuario = models.ForeignKey('Usuario', on_delete=models.SET_NULL, null=True, blank=True)
    id_transaccion = models.OneToOneField('Transaccion', on_delete=models.CASCADE)
    comentario = models.TextField(null=True, blank=True)
    estado = models.CharField(max_length=15, default='Pendiente')
    es_fraude = models.BooleanField(default=False)
    fecha = models.DateTimeField(auto_now_add=True)
    score_riesgo = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    explicabilidad = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"Incidente {self.id_incidente} - Estado: {self.estado}"


class Configuracion(models.Model):
    umbral_score = models.IntegerField(default=70)
    actualizado_en = models.DateTimeField(auto_now=True)
    actualizado_por = models.ForeignKey('Usuario', on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f"Umbral {self.umbral_score}%"
