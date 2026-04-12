# Mejora Integral de MyFraudLock

Análisis completo del repositorio y plan de mejora en **seguridad, calidad de código, UX/UI y DevOps**.

## Resumen del Proyecto Actual

MyFraudLock es un sistema Django + DRF para detección de fraude con ML (DNN + SHAP), con:
- Autenticación custom basada en sesiones (⚠️ contraseñas en texto plano)
- 3 roles: ADMIN, ANALISTA, EJECUTIVO
- Dashboard con Chart.js, listado de incidentes con polling, auditoría individual/lote
- Modelo DNN (Keras) con explicabilidad SHAP
- Frontend con Bootstrap 5 y estilos inline

---

## User Review Required

> [!CAUTION]
> **Contraseñas almacenadas en texto plano** — Actualmente `models.py` guarda passwords como `CharField` sin hash. La mejora migrará a `make_password` / `check_password` de Django. Las contraseñas existentes en la BD dejarán de funcionar hasta re-crear los usuarios o hacer un script de migración de datos.

> [!WARNING]
> **SECRET_KEY hardcodeada** — `settings.py` tiene `dev-secret-key-change-me` como fallback. Se moverá a `.env` con `python-decouple`.

> [!IMPORTANT]
> **Mejora visual significativa** — El frontend pasará de Bootstrap estándar a un diseño dark-mode premium con glassmorphism, animaciones y tipografía moderna. El layout base cambiará completamente.

---

## Proposed Changes

### 🔒 Seguridad

#### [MODIFY] [models.py](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/api/models.py)
- Agregar métodos `set_password()` y `check_password()` utilizando `django.contrib.auth.hashers`
- Agregar campo `username` como `unique=True` (actualmente no lo es)

#### [MODIFY] [views.py](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/api/views.py)
- Cambiar `usuario.password != password` por `check_password(password, usuario.password)`
- Cambiar `password=password` en register por `set_password()`
- Manejar `int()` y `float()` conversions con try/except en `configuracion_front`
- Usar `{% url %}` en lugar de hardcoded `/api/incidentes/...` en templates

#### [MODIFY] [settings.py](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/MyFraudLock/settings.py)
- Integrar `python-decouple` para `SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS`, `DATABASE_URL`
- Agregar `SECURE_BROWSER_XSS_FILTER`, `X_CONTENT_TYPE_NOSNIFF`, `SESSION_COOKIE_HTTPONLY`
- Agregar `LOGGING` config para capturar warnings/errors

#### [NEW] [.env.example](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/.env.example)
- Template de variables de entorno

---

### 🧹 Calidad de Código

#### [MODIFY] [views.py](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/api/views.py)
- Extraer decoradores (`usuario_login_required`, `require_roles`) a un archivo separado `decorators.py`
- Extraer lógica de auditoría a funciones helper para reducir la complejidad de las views
- Agregar docstrings a todas las funciones
- Eliminar import innecesario: `from django.contrib.auth.models import User` en models.py

#### [NEW] [decorators.py](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/api/decorators.py)
- `usuario_login_required` y `require_roles` extraídos de views.py

#### [MODIFY] [admin.py](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/api/admin.py)
- Registrar todos los modelos (Transaccion, Incidente, Configuracion)
- Agregar `list_display`, `list_filter`, `search_fields` para cada modelo
- Ocultar campo password en el admin de Usuario

#### [MODIFY] [serializers.py](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/api/serializers.py)
- Excluir `password` del `UsuarioSerializer` (actualmente expone contraseñas vía API REST)
- Agregar validaciones de campos

#### [NEW] [tests.py](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/api/tests.py) (reescribir)
- Tests para login/register/logout
- Tests para decoradores de permisos
- Tests para API endpoints

---

### 🎨 Frontend Premium (UI/UX)

#### [MODIFY] [base.html](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/templates/api/base.html)
- **Rediseño completo del layout**:
  - Dark sidebar nav con gradientes y glassmorphism
  - Google Fonts (Inter)
  - Micro-animaciones y transiciones
  - Footer con información del sistema
  - Mejor sistema de toasts con animaciones
  - Variables CSS centralizadas (eliminar estilos inline dispersos)
  - Responsive sidebar → hamburger en móvil

#### [MODIFY] [login.html](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/templates/api/login.html)
- Diseño standalone (sin usar base.html con navbar)
- Fondo animado con partículas/gradientes  
- Card con glassmorphism
- Animaciones de entrada

#### [MODIFY] [register.html](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/templates/api/register.html)
- Mismo estilo premium que login
- Indicador de fortaleza de contraseña visual

#### [MODIFY] [inicio.html](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/templates/api/inicio.html)
- KPIs con gradientes y iconos animados
- Cards hover con efecto 3D sutil
- Hero section mejorado

#### [MODIFY] [dashboard.html](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/templates/api/dashboard.html)
- Cards de KPI con gradientes brand
- Gráficos con tema oscuro
- Mejor tabla de últimos incidentes

#### [MODIFY] [incidentes.html](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/templates/api/incidentes.html)
- Limpiar CSS duplicado (`.badge-risk-*` definido 3+ veces)
- Tabla con diseño consistente con el nuevo tema

#### [MODIFY] [incidente_detalle.html](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/templates/api/incidente_detalle.html)
- Limpiar HTML mal anidado (mensajes duplicados, divs sin cerrar correctamente)
- Mejorar visualización de explicabilidad con gráfico de barras interactivo (SHAP waterfall)

#### [MODIFY] [auditoria.html](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/templates/api/auditoria.html)
- Mover `<style>` dentro del `{% block content %}` (actualmente está fuera)
- Gauge/velocímetro visual para el score

#### [MODIFY] [configuracion.html](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/templates/api/configuracion.html)
- Slider visual para el umbral además del input numérico

#### [MODIFY] [ayuda.html](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/templates/api/ayuda.html)
- Acordeones para FAQ
- Mejor organización visual

---

### 🔧 DevOps / Infraestructura

#### [NEW] [.gitignore](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/.gitignore)
- db.sqlite3, __pycache__, .env, staticfiles/, .idea/, *.pyc

#### [MODIFY] [requirements.txt](file:///home/loktar/Documentos/Github%20Projects/MyFraudLock/requirements.txt)
- Agregar `python-decouple`
- Pin versions de dependencias ML para reproducibilidad

---

## Open Questions

> [!IMPORTANT]
> 1. **¿Quieres que migre las contraseñas existentes?** Si ya tienes usuarios en `db.sqlite3`, puedo crear un script de migración que hashee las contraseñas actuales. ¿O prefieres recrear los usuarios?

> [!IMPORTANT]  
> 2. **¿Quieres que la mejora visual sea dark mode o light mode premium?** Recomiendo dark mode (sidebar oscura + contenido claro) por ser más moderno para dashboards de seguridad, pero puedo hacer light premium si prefieres.

> [!IMPORTANT]
> 3. **¿Hay alguna funcionalidad adicional que quieras agregar?** (ej: gestión de usuarios desde el panel, exportación de reportes PDF, notificaciones por email, etc.)

---

## Verification Plan

### Automated Tests
- `python manage.py test api` — ejecutar tests unitarios nuevos
- `python manage.py check --deploy` — verificar configuración de producción
- `python manage.py makemigrations --check` — verificar migraciones pendientes

### Manual Verification
- Verificar login/register/logout funciona correctamente
- Verificar que la API REST no expone contraseñas
- Navegar todas las páginas y verificar el nuevo diseño responsive
- Probar en modo móvil (viewport < 768px)
