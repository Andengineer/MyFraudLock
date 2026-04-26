from rest_framework import serializers
from .models import Usuario, Transaccion, Incidente, Configuracion


class UsuarioSerializer(serializers.ModelSerializer):
    """Serializer público — nunca expone la contraseña."""
    rol = serializers.ChoiceField(choices=Usuario.Roles.choices, required=False)

    class Meta:
        model = Usuario
        exclude = ('password',)


class TransaccionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Transaccion
        fields = [
            "id_transaccion", "importe", "fecha",
            "card_brand", "card_type", "issuer_bank",
            "payment_channel", "eci_code", "num_installments",
            "customer_region", "city_population",
            "is_new_customer", "days_since_first_purchase",
            "avg_historical_amount",
            "category", "num_items", "has_discount",
            "previous_failed_attempts",
            "session_duration_minutes", "interaction_velocity",
            "device_telemetry_1", "device_telemetry_2",
            "device_telemetry_3", "device_telemetry_4",
            "device_telemetry_5"
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
