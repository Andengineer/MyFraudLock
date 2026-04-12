# Definición de Requisitos del Sistema — MyFraudLock

## 1. Requisitos Funcionales (RF)

La siguiente tabla presenta la especificación formal de los requisitos funcionales del sistema MyFraudLock, identificados a partir del análisis de los módulos implementados y de las necesidades operativas de un sistema antifraude transaccional en producción. Los requisitos se encuentran clasificados por prioridad de implementación (Alta, Media, Baja) conforme a su impacto en la continuidad operativa y la seguridad de la información.

### 1.1 Módulo de Autenticación y Gestión de Sesiones

| ID | Requisito Funcional | Prioridad |
|----|---------------------|-----------|
| **RF-01** | El sistema debe permitir el inicio de sesión de usuarios registrados mediante nombre de usuario o correo electrónico, junto con una contraseña previamente almacenada de manera segura utilizando el algoritmo de hashing PBKDF2. | Alta |
| **RF-02** | El sistema debe validar que el usuario se encuentre en estado activo antes de autorizar el acceso. Si el usuario se encuentra desactivado, el sistema denegará la autenticación y mostrará un mensaje informativo. | Alta |
| **RF-03** | El sistema debe crear una sesión segura del lado del servidor (session-based authentication) al autenticar exitosamente al usuario, almacenando el identificador del usuario, su rol y su nombre de usuario en la sesión HTTP. | Alta |
| **RF-04** | El sistema debe ofrecer la opción "Recordar sesión", que extiende la vigencia de la cookie de sesión a catorce (14) días calendario. Si no se selecciona esta opción, la sesión expirará al cerrar el navegador. | Media |
| **RF-05** | El sistema debe permitir el cierre de sesión de forma segura, invalidando completamente la sesión activa del servidor mediante la función `flush()` y redirigiendo al usuario a la pantalla de inicio de sesión. | Alta |
| **RF-06** | El sistema debe permitir el registro de nuevos usuarios requiriendo los campos obligatorios: nombre de usuario (único), correo electrónico (único), contraseña (mínimo 6 caracteres) y confirmación de contraseña. | Alta |
| **RF-07** | El sistema debe validar durante el registro que el nombre de usuario y el correo electrónico no se encuentren previamente registrados en la base de datos, informando al usuario en caso de duplicidad. | Alta |
| **RF-08** | El sistema debe redirigir automáticamente a los usuarios ya autenticados que intenten acceder a las vistas de inicio de sesión o registro hacia la página principal del sistema. | Baja |

### 1.2 Módulo de Control de Acceso Basado en Roles (RBAC)

| ID | Requisito Funcional | Prioridad |
|----|---------------------|-----------|
| **RF-09** | El sistema debe implementar un esquema de control de acceso basado en tres roles jerárquicos: Administrador (ADMIN), Analista de Fraude (ANALISTA) y Ejecutivo (EJECUTIVO), donde el rol de Administrador hereda acceso total a todos los módulos. | Alta |
| **RF-10** | El sistema debe restringir el acceso al módulo de Dashboard exclusivamente a los usuarios con rol Ejecutivo o Administrador. | Alta |
| **RF-11** | El sistema debe restringir el acceso al módulo de Gestión de Incidentes (listado y detalle) exclusivamente a los usuarios con rol Analista o Administrador. | Alta |
| **RF-12** | El sistema debe restringir el acceso al módulo de Simulación de Riesgo (individual y por lote) a los usuarios con rol Analista, Ejecutivo o Administrador. | Alta |
| **RF-13** | El sistema debe restringir el acceso al módulo de Configuración del Sistema exclusivamente a los usuarios con rol Ejecutivo o Administrador. | Alta |
| **RF-14** | El sistema debe restringir el acceso al módulo de Gestión de Usuarios exclusivamente a los usuarios con rol Administrador. | Alta |
| **RF-15** | En caso de que un usuario autenticado intente acceder a un módulo para el cual no posee los permisos requeridos, el sistema debe redirigirlo a la página de inicio y mostrar un mensaje de error indicando la denegación de acceso. | Media |

### 1.3 Módulo de Recepción y Procesamiento de Transacciones

| ID | Requisito Funcional | Prioridad |
|----|---------------------|-----------|
| **RF-16** | El sistema debe exponer un endpoint REST (POST `/api/transacciones/`) que permita la recepción de transacciones financieras en formato JSON con los campos: importe, categoría del comercio, estado/región geográfica, género, edad y población de la ciudad. | Alta |
| **RF-17** | El sistema debe procesar cada transacción recibida a través del modelo de Deep Learning (Red Neuronal Profunda basada en Keras) embebido en el servidor, generando un score de riesgo en porcentaje (0-100). | Alta |
| **RF-18** | El sistema debe derivar automáticamente las variables temporales (hora, día de la semana, mes, indicador de fin de semana) a partir de la marca temporal de recepción de la transacción para alimentar el modelo predictivo. | Alta |
| **RF-19** | El sistema debe aplicar la transformación logarítmica `log1p` al campo de importe como parte del preprocesamiento de variables cuantitativas antes de la inferencia del modelo neuronal. | Media |
| **RF-20** | El sistema debe generar automáticamente un vector de explicabilidad mediante SHAP (SHapley Additive exPlanations) que identifique los factores más influyentes en la predicción del modelo para cada transacción procesada. | Alta |

### 1.4 Módulo de Gestión de Incidentes

| ID | Requisito Funcional | Prioridad |
|----|---------------------|-----------|
| **RF-21** | El sistema debe crear automáticamente un registro de Incidente con estado "Pendiente" cuando el score de riesgo de una transacción iguale o supere el umbral de tolerancia configurado globalmente. | Alta |
| **RF-22** | El sistema debe presentar una vista de listado de incidentes ordenados por fecha descendente, mostrando el identificador, score de riesgo, estado actual y fecha de creación de cada incidente. | Alta |
| **RF-23** | El sistema debe permitir al usuario Analista acceder al detalle de un incidente específico, visualizando la información completa de la transacción asociada, el score de riesgo, los factores de explicabilidad SHAP y el estado actual. | Alta |
| **RF-24** | El sistema debe permitir al usuario Analista dictaminar un incidente como "Fraude confirmado" o "Falso positivo", exigiendo la redacción obligatoria de un comentario de justificación antes de guardar el cambio de estado. | Alta |
| **RF-25** | El sistema debe registrar de manera inmutable el identificador del usuario que gestionó cada incidente, creando una pista de auditoría permanente que vincule al analista responsable con la decisión tomada. | Alta |
| **RF-26** | El sistema debe exponer un endpoint REST para polling (`GET /api/incidentes/since/?after_id=N`) que retorne los incidentes creados después de un identificador dado, permitiendo la actualización en tiempo casi real del listado sin recargar la página. | Media |

### 1.5 Módulo de Simulación de Riesgo

| ID | Requisito Funcional | Prioridad |
|----|---------------------|-----------|
| **RF-27** | El sistema debe permitir la simulación de riesgo individual de una transacción ingresando manualmente sus datos (importe, categoría, estado, género, edad y población de la ciudad), retornando el score de riesgo y los factores de explicabilidad sin persistir el resultado en la base de datos. | Alta |
| **RF-28** | El sistema debe permitir la simulación de riesgo masiva por lote mediante la carga de un archivo CSV con las columnas requeridas (importe, category, state, gender, age, city_pop), procesando cada fila a través del modelo de inferencia y presentando los resultados ordenados por score de riesgo descendente. | Alta |
| **RF-29** | El sistema debe validar que el archivo CSV de simulación por lote contenga las cabeceras requeridas, informando al usuario inmediatamente si el formato es incorrecto. | Media |
| **RF-30** | El sistema debe manejar de forma robusta las filas del archivo CSV que contengan datos inválidos (valores no numéricos, campos vacíos), omitiendo las filas erróneas y continuando el procesamiento de las restantes. | Media |

### 1.6 Módulo de Administración de Usuarios

| ID | Requisito Funcional | Prioridad |
|----|---------------------|-----------|
| **RF-31** | El sistema debe permitir al Administrador visualizar el listado completo de usuarios registrados, junto con estadísticas de resumen (usuarios activos, inactivos y cantidad de administradores). | Alta |
| **RF-32** | El sistema debe permitir al Administrador crear nuevos usuarios asignándoles nombre de usuario, correo electrónico, teléfono opcional, contraseña (mínimo 6 caracteres) y rol del sistema. | Alta |
| **RF-33** | El sistema debe permitir al Administrador editar los datos de un usuario existente (correo electrónico, teléfono, rol y estado activo/inactivo), sin modificar la contraseña en este flujo. | Media |
| **RF-34** | El sistema debe permitir al Administrador activar o desactivar usuarios de forma lógica (soft delete), impidiendo que un Administrador se desactive a sí mismo para evitar la pérdida de acceso administrativo. | Alta |
| **RF-35** | El sistema debe permitir al Administrador restablecer la contraseña de cualquier usuario, aplicando la validación de longitud mínima de 6 caracteres y almacenando la nueva contraseña de forma hasheada. | Alta |

### 1.7 Módulo de Configuración del Sistema

| ID | Requisito Funcional | Prioridad |
|----|---------------------|-----------|
| **RF-36** | El sistema debe permitir la configuración dinámica del umbral de score de riesgo (valor entero entre 0 y 100), el cual determina el punto de corte a partir del cual una transacción genera automáticamente un incidente. | Alta |
| **RF-37** | El sistema debe permitir activar o desactivar el envío de notificaciones por correo electrónico de manera global desde el módulo de configuración. | Media |
| **RF-38** | El sistema debe registrar la fecha de última actualización de la configuración y el usuario responsable del cambio, manteniendo la trazabilidad de modificaciones al umbral operativo. | Media |

### 1.8 Módulo de Notificaciones por Correo Electrónico

| ID | Requisito Funcional | Prioridad |
|----|---------------------|-----------|
| **RF-39** | El sistema debe enviar una notificación por correo electrónico a todos los Analistas y Administradores activos cuando se genere un nuevo incidente, incluyendo el identificador del incidente y el score de riesgo. | Media |
| **RF-40** | El sistema debe enviar una notificación por correo electrónico a todos los Ejecutivos y Administradores activos cuando un incidente sea dictaminado como "Fraude confirmado", incluyendo los datos relevantes del incidente. | Media |
| **RF-41** | El sistema debe verificar que las notificaciones por correo electrónico se encuentren habilitadas en la configuración global antes de intentar el envío, respetando el estado del parámetro `notificaciones_email`. | Media |

### 1.9 Módulo de Exportación de Reportes PDF

| ID | Requisito Funcional | Prioridad |
|----|---------------------|-----------|
| **RF-42** | El sistema debe permitir la generación y descarga de un reporte PDF del Dashboard Ejecutivo que incluya los KPIs de incidentes (pendientes, fraude confirmado, falsos positivos) y una tabla con los últimos 20 incidentes registrados. | Media |
| **RF-43** | El sistema debe permitir la generación y descarga de un reporte PDF del listado completo de incidentes, incluyendo identificador, score de riesgo, estado y fecha de cada registro. | Media |
| **RF-44** | El sistema debe permitir la generación y descarga de un reporte PDF de detalle de un incidente individual, incluyendo los datos del incidente, la transacción asociada y los factores de explicabilidad SHAP organizados en formato tabular. | Media |

### 1.10 Módulo de Dashboard Ejecutivo

| ID | Requisito Funcional | Prioridad |
|----|---------------------|-----------|
| **RF-45** | El sistema debe presentar un panel de control (Dashboard) con indicadores clave de rendimiento (KPIs): total de incidentes pendientes, total de fraudes confirmados y total de falsos positivos. | Alta |
| **RF-46** | El sistema debe presentar gráficos interactivos que muestren la distribución de fraudes confirmados por región geográfica (top 10 + agrupación "Otros") y por categoría de comercio. | Media |
| **RF-47** | El sistema debe presentar un histograma de distribución de scores de riesgo segmentado en cinco rangos (0–20, 20–40, 40–60, 60–80, 80–100). | Media |
| **RF-48** | El sistema debe presentar un gráfico de tendencia temporal que muestre la evolución diaria de incidentes (Pendientes, Fraude confirmado, Falso positivo) durante los últimos treinta (30) días calendario. | Media |

---

## 2. Requisitos No Funcionales (RNF)

La siguiente tabla define los requisitos no funcionales del sistema, clasificados por categoría según el estándar ISO/IEC 25010:2011 de calidad de producto de software.

| ID | Requisito No Funcional | Categoría |
|----|------------------------|-----------|
| **RNF-01** | El tiempo de respuesta de la inferencia del modelo de Deep Learning no debe exceder los 500 milisegundos por transacción individual, medido desde la recepción de la petición POST hasta la emisión de la respuesta HTTP 201. | Rendimiento |
| **RNF-02** | El sistema debe procesar archivos CSV de simulación de riesgo por lote con un mínimo de 500 registros sin degradación perceptible de la experiencia de usuario (tiempo total inferior a 60 segundos). | Rendimiento |
| **RNF-03** | Todas las comunicaciones entre el cliente y el servidor en entorno de producción deben realizarse exclusivamente bajo protocolo HTTPS con certificación TLS 1.2 o superior, con redirección automática de tráfico HTTP. | Seguridad |
| **RNF-04** | Las contraseñas de los usuarios deben almacenarse utilizando el algoritmo de hashing PBKDF2 con salt aleatorio, proporcionado por `django.contrib.auth.hashers`, impidiendo la lectura de contraseñas en texto plano desde la base de datos. | Seguridad |
| **RNF-05** | El sistema debe implementar protección contra ataques Cross-Site Request Forgery (CSRF) en todos los formularios de mutación de datos, así como las cabeceras de seguridad `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff` y `X-XSS-Protection`. | Seguridad |
| **RNF-06** | La API REST no debe exponer campos sensibles (como contraseñas o tokens de sesión) en ninguna de sus respuestas, excluyendo explícitamente el campo `password` del serializer `UsuarioSerializer`. | Seguridad |
| **RNF-07** | La interfaz de usuario debe seguir los principios heurísticos de Nielsen, implementando un diseño Dark Mode premium con tipografía moderna (Google Fonts — Inter), glassmorphism y micro-animaciones, optimizado para operadores de seguridad en monitoreo continuo 24/7. | Usabilidad |
| **RNF-08** | El diseño de la interfaz debe ser responsivo y adaptable a dispositivos con resoluciones desde 320px (móvil) hasta 2560px (monitor ultra-wide), utilizando media queries y un sidebar colapsable en modo hamburguesa para pantallas inferiores a 768px. | Usabilidad |
| **RNF-09** | La arquitectura web del sistema debe ser stateless a nivel del servidor de aplicaciones, permitiendo el reinicio o escalamiento horizontal de los contenedores sin pérdida de información de negocio (objetivo de disponibilidad: 99.9%). | Confiabilidad |
| **RNF-10** | Los archivos estáticos del sistema (CSS, JavaScript, imágenes) deben servirse a través del middleware WhiteNoise con compresión y cache manifest, reduciendo la carga al servidor de aplicaciones y mejorando los tiempos de carga en el cliente. | Eficiencia |
| **RNF-11** | El sistema debe registrar las operaciones críticas (errores, advertencias de seguridad, notificaciones enviadas) mediante el framework de logging de Django, con salida formateada a la consola del servidor. | Mantenibilidad |
| **RNF-12** | El sistema debe utilizar variables de entorno para la configuración de parámetros sensibles (SECRET_KEY, credenciales de base de datos, credenciales SMTP), impidiendo el almacenamiento de secretos en el código fuente. | Portabilidad |
