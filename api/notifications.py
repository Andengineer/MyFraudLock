"""
Notificaciones por email para eventos de fraude.
"""
import logging

from django.conf import settings
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.html import strip_tags

from .models import Configuracion, Usuario

logger = logging.getLogger(__name__)


def _is_enabled() -> bool:
    """Retorna True si las notificaciones están habilitadas."""
    try:
        config = Configuracion.objects.get(pk=1)
        return config.notificaciones_email
    except Configuracion.DoesNotExist:
        return True  # por defecto habilitadas


def notify_new_incident(incidente):
    """
    Envía email a todos los ANALISTAS activos cuando se crea un incidente.
    """
    if not _is_enabled():
        return

    analistas = Usuario.objects.filter(
        rol__in=["ANALISTA", "ADMIN"], activo=True
    ).exclude(email="")

    if not analistas.exists():
        logger.info("No hay analistas activos para notificar.")
        return

    ctx = {
        "incidente": incidente,
        "transaccion": incidente.id_transaccion,
        "score": float(incidente.score_riesgo or 0),
    }

    html = render_to_string("api/email/nuevo_incidente.html", ctx)
    plain = strip_tags(html)

    recipients = list(analistas.values_list("email", flat=True))

    try:
        send_mail(
            subject=f"⚠️ Nuevo incidente #{incidente.id_incidente} — Score {ctx['score']:.1f}%",
            message=plain,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=recipients,
            html_message=html,
            fail_silently=True,
        )
        logger.info(
            "Email enviado a %d analista(s) por incidente #%s",
            len(recipients), incidente.id_incidente,
        )
    except Exception as e:
        logger.error("Error enviando email: %s", e)


def notify_fraud_confirmed(incidente):
    """
    Envía email a todos los GERENTES activos cuando se confirma fraude.
    """
    if not _is_enabled():
        return

    ejecutivos = Usuario.objects.filter(
        rol__in=["GERENTE", "ADMIN"], activo=True
    ).exclude(email="")

    if not ejecutivos.exists():
        logger.info("No hay ejecutivos activos para notificar.")
        return

    ctx = {
        "incidente": incidente,
        "transaccion": incidente.id_transaccion,
        "score": float(incidente.score_riesgo or 0),
    }

    html = render_to_string("api/email/fraude_confirmado.html", ctx)
    plain = strip_tags(html)

    recipients = list(ejecutivos.values_list("email", flat=True))

    try:
        send_mail(
            subject=f"🚨 Fraude confirmado — Incidente #{incidente.id_incidente}",
            message=plain,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=recipients,
            html_message=html,
            fail_silently=True,
        )
        logger.info(
            "Email enviado a %d ejecutivo(s) por fraude confirmado #%s",
            len(recipients), incidente.id_incidente,
        )
    except Exception as e:
        logger.error("Error enviando email: %s", e)
