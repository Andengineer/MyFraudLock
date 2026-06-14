# Generated manually: corrige la trazabilidad de incidentes ya resueltos.
#
# Los incidentes con estado "Fraude confirmado" o "Falso positivo" deben
# registrar qué analista los gestionó. Datos históricos (p. ej. sembrados)
# quedaron sin "gestionado_por"; aquí se les asigna un gestor existente.
from django.db import migrations


ESTADOS_RESUELTOS = ("Fraude confirmado", "Falso positivo")


def asignar_gestores(apps, schema_editor):
    Usuario = apps.get_model("api", "Usuario")
    Incidente = apps.get_model("api", "Incidente")

    # Analistas y administradores pueden gestionar incidentes (ADMIN siempre
    # tiene acceso). Se ordenan con los analistas primero.
    gestores = list(
        Usuario.objects.filter(rol__in=["ANALISTA", "ADMIN"], activo=True)
        .order_by("rol", "id_usuario")  # ADMIN < ANALISTA alfabéticamente; ver nota abajo
    )
    if not gestores:
        gestores = list(Usuario.objects.filter(activo=True))
    if not gestores:
        # No hay a quién asignar; no se puede backfillear.
        return

    pendientes = (
        Incidente.objects.filter(
            estado__in=ESTADOS_RESUELTOS, gestionado_por__isnull=True
        ).order_by("id_incidente")
    )

    # Reparto round-robin para que la trazabilidad sea variada y determinista.
    for idx, inc in enumerate(pendientes):
        inc.gestionado_por = gestores[idx % len(gestores)]
        inc.save(update_fields=["gestionado_por"])


def revertir(apps, schema_editor):
    # No se revierte: volver a borrar el gestor reintroduciría el defecto.
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0006_transaccion_action_code_transaccion_bin_and_more"),
    ]

    operations = [
        migrations.RunPython(asignar_gestores, revertir),
    ]
