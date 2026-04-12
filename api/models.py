from django.db import models
from django.contrib.auth.hashers import make_password, check_password as _check_pw


class Usuario(models.Model):
    """Usuario del sistema con rol y autenticación propia."""

    class Roles(models.TextChoices):
        ADMIN = 'ADMIN', 'Administrador'
        ANALISTA = 'ANALISTA', 'Analista de Fraude'
        EJECUTIVO = 'EJECUTIVO', 'Ejecutivo (solo lectura)'

    id_usuario = models.AutoField(primary_key=True)
    username = models.CharField(max_length=100, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128)  # almacena hash
    telefono = models.CharField(max_length=15, null=True, blank=True)
    activo = models.BooleanField(default=True)
    rol = models.CharField(
        max_length=15,
        choices=Roles.choices,
        default=Roles.ANALISTA,
        db_index=True,
    )

    def set_password(self, raw_password: str):
        """Hashea y almacena la contraseña."""
        self.password = make_password(raw_password)

    def check_password(self, raw_password: str) -> bool:
        """Compara una contraseña en texto plano con el hash almacenado."""
        return _check_pw(raw_password, self.password)

    def __str__(self):
        return self.username


class Transaccion(models.Model):
    """Transacción financiera que alimenta el modelo de ML."""

    id_transaccion = models.AutoField(primary_key=True)
    importe = models.DecimalField(max_digits=12, decimal_places=2)
    fecha = models.DateTimeField(auto_now_add=True)

    category = models.CharField(max_length=32)
    state = models.CharField(max_length=32)
    gender = models.CharField(max_length=1)   # 'm' / 'f'
    age = models.PositiveSmallIntegerField()
    city_pop = models.PositiveIntegerField()

    def __str__(self):
        return f"Tx #{self.id_transaccion}"


class Incidente(models.Model):
    """Incidente generado cuando el score supera el umbral."""

    id_incidente = models.AutoField(primary_key=True)
    gestionado_por = models.ForeignKey(
        'Usuario', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='incidentes_gestionados'
    )
    id_transaccion = models.OneToOneField(
        'Transaccion', on_delete=models.CASCADE
    )
    comentario = models.TextField(null=True, blank=True)
    estado = models.CharField(max_length=20, default='Pendiente')
    es_fraude = models.BooleanField(default=False)
    fecha = models.DateTimeField(auto_now_add=True)
    score_riesgo = models.DecimalField(
        max_digits=5, decimal_places=2, null=True, blank=True
    )
    explicabilidad = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"Incidente {self.id_incidente} - Estado: {self.estado}"


class Configuracion(models.Model):
    """Configuración global del sistema (umbral, etc.)."""

    umbral_score = models.IntegerField(default=70)
    notificaciones_email = models.BooleanField(
        default=True,
        help_text="Enviar emails cuando se detecta fraude",
    )
    actualizado_en = models.DateTimeField(auto_now=True)
    actualizado_por = models.ForeignKey(
        'Usuario', on_delete=models.SET_NULL, null=True, blank=True
    )

    def __str__(self):
        return f"Umbral {self.umbral_score}%"
