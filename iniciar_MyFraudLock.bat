@echo off
REM ============================================================
REM  MyFraudLock  -  Instalador y ejecutor automatico (Windows)
REM  Doble click para:
REM    1) Verificar Python
REM    2) Crear/activar entorno virtual .venv
REM    3) Instalar dependencias (requirements.txt)
REM    4) Aplicar migraciones de base de datos
REM    5) Crear usuario administrador por defecto (si no existe)
REM    6) Iniciar el servidor Django en http://127.0.0.1:8000
REM ============================================================

setlocal EnableDelayedExpansion
cd /d "%~dp0"

title MyFraudLock - Instalador y Servidor

echo.
echo ================================================================
echo                  MyFraudLock  -  Inicializador
echo ================================================================
echo.

REM ---------- 1) Verificar que Python esta instalado ----------
echo [1/6] Verificando instalacion de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python no esta instalado o no esta en el PATH.
    echo         Descargalo desde https://www.python.org/downloads/
    echo         Asegurate de marcar la casilla "Add Python to PATH".
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo        %%v detectado.
echo.

REM ---------- 2) Crear / activar el entorno virtual ----------
echo [2/6] Preparando entorno virtual (.venv)...
if not exist ".venv\Scripts\activate.bat" (
    echo        Creando entorno virtual nuevo...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
    set "PRIMERA_INSTALACION=1"
) else (
    echo        Entorno virtual ya existe.
    set "PRIMERA_INSTALACION=0"
)
call ".venv\Scripts\activate.bat"
echo        Entorno virtual activado.
echo.

REM ---------- 3) Instalar dependencias ----------
echo [3/6] Instalando dependencias de requirements.txt...
echo        (esto puede demorar varios minutos la primera vez)
python -m pip install --upgrade pip --quiet
if "!PRIMERA_INSTALACION!"=="1" (
    python -m pip install -r requirements.txt
) else (
    python -m pip install -r requirements.txt --quiet
)
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias.
    pause
    exit /b 1
)
echo        Dependencias instaladas correctamente.
echo.

REM ---------- 4) Aplicar migraciones ----------
echo [4/6] Aplicando migraciones de base de datos...
python manage.py migrate --noinput
if errorlevel 1 (
    echo [ERROR] Fallo al aplicar migraciones.
    pause
    exit /b 1
)
echo        Base de datos lista.
echo.

REM ---------- 5) Crear usuario administrador por defecto ----------
echo [5/6] Verificando usuario administrador por defecto...
python -c "import os,django;os.environ.setdefault('DJANGO_SETTINGS_MODULE','MyFraudLock.settings');django.setup();from api.models import Usuario;u,c=Usuario.objects.get_or_create(username='admin',defaults={'email':'admin@myfraudlock.local','rol':'ADMIN','activo':True});  (u.set_password('Admin123') or u.save()) if c else None;print('   -> Credenciales: admin / Admin123 (cambiar al primer inicio de sesion)' if c else '   -> Usuario admin ya existe.')"
echo.

REM ---------- 6) Iniciar el servidor ----------
echo [6/6] Iniciando el servidor de desarrollo Django...
echo.
echo ================================================================
echo   Abre tu navegador en:  http://127.0.0.1:8000/api/login/
echo   Usuario: admin     Clave: Admin123
echo   Para detener el servidor presiona  Ctrl + C  en esta ventana.
echo ================================================================
echo.

REM Abrir el navegador automaticamente despues de 4 segundos
start "" /B cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:8000/api/login/"

python manage.py runserver 127.0.0.1:8000

echo.
echo Servidor detenido. Presiona una tecla para cerrar.
pause >nul
endlocal
