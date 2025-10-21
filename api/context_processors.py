from .models import Usuario

def current_usuario(request):
    uid = request.session.get('usuario_id')
    usuario = None
    if uid:
        try:
            usuario = Usuario.objects.get(pk=uid, activo=True)
        except Usuario.DoesNotExist:
            pass
    return {"current_usuario": usuario}
