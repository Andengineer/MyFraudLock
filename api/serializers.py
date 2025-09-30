from rest_framework import serializers
from .models import Usuario, Transaccion, Incidente, Configuracion

class UsuarioSerializer(serializers.ModelSerializer):
    rol = serializers.ChoiceField(choices=Usuario.Roles.choices, required=False)
    class Meta:
        model = Usuario
        fields = '__all__'

class TransaccionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Transaccion
        fields = [
            "id_transaccion", "importe", "fecha",
            "category", "state", "gender",
            "age", "city_pop",
        ]

class IncidenteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Incidente
        fields = '__all__'

    def validate_id_transaccion(self, value):
        if Incidente.objects.filter(id_transaccion=value).exists():
            raise serializers.ValidationError("Esta transacción ya tiene un incidente registrado.")
        return value

class ConfiguracionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Configuracion
        fields = '__all__'

