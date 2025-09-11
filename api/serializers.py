from rest_framework import serializers
from .models import Usuario
from .models import Transaccion
from .models import Incidente

class UsuarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = Usuario
        fields = '__all__'

class TransaccionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Transaccion
        fields = '__all__'

class IncidenteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Incidente
        fields = '__all__'

    def validate_id_transaccion(self, value):
        if Incidente.objects.filter(id_transaccion=value).exists():
            raise serializers.ValidationError("Esta transacción ya tiene un incidente registrado.")
        return value
