from django.db import models

class Usuario(models.Model):
    id_usuario = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length=100, null=False)
    email = models.EmailField(unique=True, null=False)
    password = models.CharField(max_length=255, null=False)
    telefono = models.CharField(max_length=20, null=True, blank=True)
    activo = models.BooleanField(default=True)

    def __str__(self):
        return self.nombre
class Transaccion(models.Model):
    id_transaccion = models.AutoField(primary_key=True)
    importe = models.DecimalField(max_digits=10, decimal_places=3, null=False)
    metodo_pago = models.CharField(max_length=50, null=False)
    direccion_envio = models.CharField(max_length=100, null=False)
    fecha = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Transacción {self.id_transaccion} - {self.metodo_pago} - {self.importe}"
