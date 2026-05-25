from rest_framework import serializers
from .models import Usuario, Transaccion, Incidente, Configuracion


class UsuarioSerializer(serializers.ModelSerializer):
    """Serializer público — nunca expone la contraseña."""
    rol = serializers.ChoiceField(choices=Usuario.Roles.choices, required=False)

    class Meta:
        model = Usuario
        exclude = ('password',)


class TransaccionSerializer(serializers.ModelSerializer):
    """
    Serializer para transacciones.

    Solo acepta datos que el comercio/pasarela envía en el momento de la compra.
    Los ratios financieros son CALCULADOS por el sistema y NO se aceptan por API.
    """
    # Opcionales con defaults razonables
    payment_channel = serializers.CharField(default="pago web", required=False)
    eci_code = serializers.IntegerField(default=5, required=False)
    num_installments = serializers.IntegerField(default=0, required=False)
    num_items = serializers.IntegerField(default=1, required=False)
    discount_amount = serializers.DecimalField(max_digits=12, decimal_places=2, default=0, required=False)
    currency = serializers.CharField(default="PEN", required=False)
    email_domain = serializers.CharField(default="gmail.com", required=False)
    bin = serializers.CharField(default="404700", required=False)
    last_4_digits = serializers.CharField(default="0000", required=False)
    wallet_yape = serializers.CharField(default="no", required=False)
    wallet_plin = serializers.CharField(default="no", required=False)
    # Internos del sistema — defaults
    transaction_status = serializers.CharField(default="liquidada", required=False)
    action_code = serializers.IntegerField(default=6, required=False)
    denial_reason = serializers.CharField(default="unknown", required=False)

    class Meta:
        model = Transaccion
        fields = [
            "id_transaccion", "importe", "fecha",
            # Tarjeta / Pago
            "card_brand", "card_type", "issuer_bank",
            "payment_channel", "eci_code", "num_installments",
            "bin", "last_4_digits",
            # Cliente
            "customer_region", "email_domain",
            # Pedido
            "product_category", "num_items",
            "discount_amount", "currency",
            # Internos
            "transaction_status", "action_code", "denial_reason",
            # Wallets
            "wallet_yape", "wallet_plin",
        ]
        read_only_fields = ["id_transaccion", "fecha"]
        # NOTA: ratio_aar, ratio_cmr, etc. NO están en fields
        # → el sistema los calcula y guarda directamente en el modelo


class TransaccionReadSerializer(serializers.ModelSerializer):
    """Serializer de lectura — incluye los ratios calculados."""
    class Meta:
        model = Transaccion
        fields = [
            "id_transaccion", "importe", "fecha",
            "card_brand", "card_type", "issuer_bank",
            "payment_channel", "eci_code", "num_installments",
            "bin", "last_4_digits",
            "customer_region", "email_domain",
            "product_category", "num_items",
            "discount_amount", "currency",
            "transaction_status", "action_code", "denial_reason",
            "wallet_yape", "wallet_plin",
            # Ratios (solo lectura, calculados por el sistema)
            "ratio_aar", "ratio_cmr", "ratio_asi",
            "ratio_vrr", "ratio_dar", "ratio_csi", "ratio_dpe",
        ]


class IncidenteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Incidente
        fields = '__all__'

    def validate_id_transaccion(self, value):
        if Incidente.objects.filter(id_transaccion=value).exists():
            raise serializers.ValidationError(
                "Esta transacción ya tiene un incidente registrado."
            )
        return value


class ConfiguracionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Configuracion
        fields = '__all__'
