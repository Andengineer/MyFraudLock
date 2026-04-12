"""
Decoradores de autenticación y autorización para vistas HTML.
"""
from functools import wraps
from django.shortcuts import redirect
from django.contrib import messages
from django.urls import reverse


def usuario_login_required(view_func):
    """Redirige a login si no hay sesión activa."""
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if not request.session.get('usuario_id'):
            next_url = request.get_full_path()
            return redirect(f"{reverse('login')}?next={next_url}")
        return view_func(request, *args, **kwargs)
    return _wrapped


def require_roles(*allowed_roles):
    """
    Restringe acceso por rol.  ADMIN siempre pasa.
    Uso: @usuario_login_required
         @require_roles('ANALISTA', 'EJECUTIVO')
    """
    def deco(view_func):
        @wraps(view_func)
        def _wrapped(request, *args, **kwargs):
            rol = request.session.get('usuario_rol')
            if not rol:
                return redirect(f"{reverse('login')}?next={request.get_full_path()}")
            if rol == 'ADMIN' or rol in allowed_roles:
                return view_func(request, *args, **kwargs)
            messages.error(request, "No tienes permisos para acceder a este módulo.")
            return redirect('inicio')
        return _wrapped
    return deco
