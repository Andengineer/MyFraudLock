import pandas as pd

# ── 1. HISTORIAS DE USUARIO ────────────────────────────
hus = [
    # Autenticación y Autorización
    ("HU0001", "Administrador", "crear usuarios con roles definidos", "administrar el acceso al sistema antifraude de manera segura", "Validación de creación", "HU0001-1", "El Admin ingresa datos", "Crea una cuenta", "Muestra éxito"),
    ("HU0002", "Ejecutivo", "iniciar sesión de forma segura y encriptada", "proteger los datos transaccionales sensibles", "Login correcto", "HU0002-1", "El Ejecutivo ingresa su email", "Hace submit", "Redirecciona al dashboard"),
    ("HU0003", "Analista", "recuperar contraseña si la olvido", "no perder el acceso de auditoría", "Recupero bloqueado", "HU0003-1", "Analista intenta cambiar pass", "Usa botón 'Olvidé contraseña'", "Desencadena fluxo de correo SMS"),
    ("HU0004", "Administrador", "desactivar usuarios inactivos o despedidos", "cortar el acceso inmediato al sistema", "Desactivación exitosa", "HU0004-1", "El Admin ubica al usuario", "Da clic en 'Desactivar', activo=False", "Usuario no puede loguearse"),
    ("HU0005", "Analista", "ser bloqueado del panel de configuración", "evitar que cambie las reglas del negocio de ML", "Control de Rutas", "HU0005-1", "Analista entra a /configuracion/", "Intenta acceder", "Es redireccionado con error de permisos"),

    # Gestión de Incidentes
    ("HU0006", "Analista", "ver un listado paginado de incidentes sospechosos", "encontrar rápidamente las alertas de fraude del día", "Listado correcto", "HU0006-1", "Analista abre módulo incidentes", "Visualiza la tabla", "Aparecen 10 filas por página"),
    ("HU0007", "Analista", "ver el semáforo y score de riesgo de cada incidente", "priorizar qué investigar primero", "Color condicional", "HU0007-1", "Hay un incidente de 95% de fraude", "Se lista en tabla", "Tiene un badge de etiqueta roja"),
    ("HU0008", "Analista", "gestionar un incidente pendiente a fraude confirmado", "alertar bloquear la tarjeta y retener fondo", "Cambio estado Fraude", "HU0008-1", "El incidente está pendiente", "Selecciona Fraude Confirmado y Guarda", "Estado cambia y bloquea la TX"),
    ("HU0009", "Analista", "gestionar un incidente a falso positivo", "liberar el fondo de un cliente legítimo bloqueado erróneamente", "Cambio estado Limpio", "HU0009-1", "El incidente está pendiente", "Selecciona Falso Positivo y Guarda", "Estado cambia y se libera el pago"),
    ("HU0010", "Analista", "descargar el detalle del incidente auditado en PDF", "adjuntar la prueba al reporte del banco o procesadora", "Exportación PDF", "HU0010-1", "El analista está viendo el ID #5", "Clic a Descargar PDF", "Se genera y descarga un .pdf"),

    # Explicabilidad y ML (XAI)
    ("HU0011", "Analista", "ver la explicabilidad SHAP de la decisión", "entender matemáticamente por qué la Inteligencia artificial bloqueó", "Renderizado SHAP", "HU0011-1", "La IA calculó el riesgo", "El analista abre el detalle", "Ve la lista de factores positivos y negativos visualmente"),
    ("HU0012", "Analista", "visualizar los factores con lenguaje de negocio en español", "no confundirse con códigos y acelerar su decisión", "Traducción de variables", "HU0012-1", "La IA arroja 'is_new_customer'", "El Front procesa", "Muestra 'Cliente Nuevo'"),
    ("HU0013", "Ejecutivo", "configurar el umbral de disparo de la red neuronal", "hacer el sistema de fraude más estricto o laxo según temporada", "Modificar Threshold", "HU0013-1", "El Ejecutivo va a la configuración", "Cambia umbral a 85%", "Nuevas transacciones con <84% no hacen incidente"),

    # Simulación de Inferencia
    ("HU0014", "Analista", "simular manualmente una transacción rellenando campos", "evaluar experimentalmente el comportamiento del modelo ML", "Simulación Individual", "HU0014-1", "Rellena el input de edad e importe", "Clic en Analizar", "Sale velocímetro con porcentaje de fraude predictivo"),
    ("HU0015", "Ejecutivo", "simular un lote CSV masivo contra la IA", "auditar históricos de ventas y encontrar fraudes pasados", "Simulación Lote", "HU0015-1", "Sube el archivo CSV", "Clic procesar lote", "El backend devuelve una tabla inferida en bulk ordenados desc"),

    # API de Transacciones (Microservicios)
    ("HU0016", "Sistema E-commerce", "mandar un payload POST transaccional con JWT/apiKey", "procesar en milisegundos si lo dejo vender o no", "API Ingestion", "HU0016-1", "Manda dict en JSON", "Recibe Response 201 o Rechazado", "Guarda en BD transaccion.importe"),
    ("HU0017", "Sistema E-commerce", "recibir respuestas 400 bad request en payload malformado", "manejar errores sin caer los servidores", "API Validation", "HU0017-1", "Falta el campo importe", "Petición POST", "Response 400 error schema validation"),
    ("HU0018", "Administrador", "ingresar por Swagger al API de integración", "ver la documentación interactiva", "OpenAPI", "HU0018-1", "Abre /api/docs/", "Se muestra UI", "Swagger UI de endpoints"),

    # Dashboard Financiero
    ("HU0019", "Ejecutivo", "visualizar tarjetas de KPI de dinero salvado vs falso", "ver el ROI de la IA en tiempo real", "KPI financieros", "HU0019-1", "Ventas de hoy marcadas a fraude", "Suma los importes de fraudes", "Muestra Dinero Salvado: S/ 4,000"),
    ("HU0020", "Ejecutivo", "visualizar gráfico en Dona de Fraude por Canal de Pago", "determinar si nos atacan más por Web o App Físicamente", "Gráfica Canales", "HU0020-1", "Agrega campo de channel", "Renderiza Canvas Chart.js", "Dona por Web, Mobile, App"),
    ("HU0021", "Ejecutivo", "visualizar grafico de Fraudes monetizados por Categorías", "entender qué inventario proteger más (Electrónica vs Hogar)", "Gráfica Categorías", "HU0021-1", "Agrupa por category / suma importe", "Pinta Chart de Barras horizontales", "La categoría con más dinero perdido gana la cima"),
    ("HU0022", "Administrador", "monitorear el ratio de fraude diario en tiempo serie", "apreciar la tendencia a lo largo de la semana en la DB", "Gráfico Temporal", "HU0022-1", "Se cuentan los incidentes diarios", "Renderiza gráfico lineal", "Curva de evolución"),
]

data_hus = []
for h in hus:
    redaccion = f"Como {h[1]} quiero {h[2]} para {h[3]}"
    data_hus.append({
        "Identificador (ID) de la Historia": h[0],
        "Rol": h[1],
        "Característica / Funcionalidad (Necesito)": h[2],
        "Razón / Resultado (Con finalidad de)": h[3],
        "REDACCION": redaccion,
        "Número (#) de Escenario": h[4],
        "Criterio de Aceptación": h[5],
        "Contexto": h[6],
        "Evento": h[7],
        "Resultado / Comportamiento esperado": h[8]
    })

df_hu = pd.DataFrame(data_hus)
df_hu.to_excel("Historias_Usuario_Listas_Para_Tesis.xlsx", index=False)

# ── 2. CASOS DE PRUEBA ──────────────────────────────────
cps = [
    # Extracción de protocolo (Añadimos ML y simulaciones)
    ("CP001", "EJECUCIÓN DE RED NEURONAL EN TIEMPO REAL (INFERENCIA DIRECTA)", "El analista ingresa un caso forzado clásico de pago falso en provincia", "1. Entrar a Simulacion\n2. Cargar S/8,000\n3. Cliente nuevo=Si, Provincia=M.Dios\n4. Pulsar 'Simular'", "Velocímetro arroja 100% Fraude de score"),
    ("CP002", "INYECCIÓN DE CSV PARA INFERENCIA MASIVA", "Se tiene CSV de 10 ventas del día pasado", "1. Entrar Simulación pestaña Lote\n2. Subir Archivo.csv\n3. Procesar", "Tabla cargada descendente. Ventas mayores en rojo"),
    ("CP003", "CÁLCULO DEL DINERO PROBABLEMENTE FRAUDULENTO Y GRÁFICO DONA", "Hay S/ 4,000 de fraude marcados en incidentes en la DB", "1. Entrar al Dashboard\n2. Sumar KPIs\n3. Ver la gráfica de Canales Operativos", "Dinero Salvado > 0 y Gráfica cargada con Chart.js"),
    ("CP004", "EMISIÓN DE LEYENDA ESPAÑOL DE ATRIBUCIONES SHAP", "Se produjo un incidente con características adversas (amount_deviation alto)", "1. Entrar al detalle\n2. Revisar la sección Factores", "Dice 'El monto supera el promedio histórico' y 'Desviación de Monto'"),
    ("CP005", "CIERRE DE FLUJO DEL ANALISTA (VEREDICTO FINAL)", "Hay un incidente listado como 'Pendiente'", "1. El Analista selecciona Falso Positivo\n2. Coloca justificación extensa\n3. Confirma el modal pop up", "El incidente desaparece de los count pendientes, pasa a Falso positivo y etiqueta en BD y el semáforo cambia a verde"),
]

data_cp = []
for c in cps:
    data_cp.append({
        "Identificador": c[0],
        "Titulo Caso Prueba": c[1],
        "Condiciones Previas": c[2],
        "Pasos Empleados": c[3],
        "Resultados Esperados Obtenidos": c[4]
    })

df_cp = pd.DataFrame(data_cp)
df_cp.to_excel("Casos_De_Prueba_Listos_Para_Tesis.xlsx", index=False)

print("✅ Todos los archivos de tesina Excel generados y compilados!")
