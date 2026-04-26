from django.db import models
from django.contrib.auth.hashers import make_password, check_password as _check_pw


class Usuario(models.Model):
    """Usuario del sistema con rol y autenticación propia."""

    class Roles(models.TextChoices):
        ADMIN = 'ADMIN', 'Administrador'
        ANALISTA = 'ANALISTA', 'Analista de Fraude'
        GERENTE = 'GERENTE', 'Gerente (solo lectura)'

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
    """Transacción financiera que alimenta el modelo de ML (Deep Learning)."""

    id_transaccion = models.AutoField(primary_key=True)
    importe = models.DecimalField(max_digits=12, decimal_places=2)
    fecha = models.DateTimeField(auto_now_add=True)

    # ── Datos de tarjeta / pago ──────────────────────────────────────
    card_brand = models.CharField(
        max_length=20, default='visa',
        help_text='Marca: visa, mastercard, amex, diners, other'
    )
    card_type = models.CharField(
        max_length=10, default='debit',
        help_text='Tipo: credit, debit'
    )
    issuer_bank = models.CharField(
        max_length=30, default='bcp',
        help_text='Banco emisor peruano'
    )
    payment_channel = models.CharField(
        max_length=15, default='web',
        help_text='Canal: web, mobile, app'
    )
    eci_code = models.PositiveSmallIntegerField(
        default=5,
        help_text='ECI de autenticación 3DS (0,2,5,6,7,11)'
    )
    num_installments = models.PositiveSmallIntegerField(
        default=0,
        help_text='Número de cuotas (0=contado)'
    )

    # ── Datos del cliente / ubicación ────────────────────────────────
    customer_region = models.CharField(
        max_length=30, default='lima',
        help_text='Región/departamento del cliente'
    )
    city_population = models.PositiveIntegerField(
        default=0,
        help_text='Población de la ciudad del cliente'
    )
    is_new_customer = models.BooleanField(
        default=True,
        help_text='¿Es primera compra del cliente?'
    )
    days_since_first_purchase = models.PositiveIntegerField(
        default=0,
        help_text='Días desde la primera compra del cliente'
    )
    avg_historical_amount = models.DecimalField(
        max_digits=12, decimal_places=2, default=0,
        help_text='Monto promedio histórico del cliente'
    )

    # ── Datos del pedido ─────────────────────────────────────────────
    category = models.CharField(
        max_length=32, default='otros',
        help_text='Categoría del producto principal'
    )
    num_items = models.PositiveSmallIntegerField(
        default=1,
        help_text='Número de ítems en el pedido'
    )
    has_discount = models.BooleanField(
        default=False,
        help_text='¿El pedido tiene descuento?'
    )
    previous_failed_attempts = models.PositiveSmallIntegerField(
        default=0,
        help_text='Intentos de pago fallidos previos a este'
    )

    # ── Biometría Conductual y Telemetría ────────────────────────────
    session_duration_minutes = models.DecimalField(
        max_digits=6, decimal_places=2, null=True, blank=True,
        help_text='Duración de la sesión en el portal (minutos)'
    )
    interaction_velocity = models.DecimalField(
        max_digits=6, decimal_places=2, null=True, blank=True,
        help_text='Velocidad de interacción'
    )
    device_telemetry_1 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    device_telemetry_2 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    device_telemetry_3 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    device_telemetry_4 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    device_telemetry_5 = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)

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
