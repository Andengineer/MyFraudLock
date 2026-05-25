# 📋 Guía de Testing — API REST MyFraudLock (DAFD-Net)

## Base URL
```
http://127.0.0.1:8000/api/
```

---

## 1. Crear Transacción (POST) — **Endpoint principal**

```
POST /api/transacciones/
Content-Type: application/json
```

> [!IMPORTANT]
> Al crear una transacción, el sistema automáticamente:
> 1. Calcula los ratios financieros (AAR, CMR, VRR, etc.) internamente
> 2. Ejecuta el modelo DAFD-Net para obtener el score
> 3. Si **score ≥ 70%**, crea un **incidente** automáticamente
>
> Solo necesitas enviar los datos que el comercio tiene en el momento de la compra.

---

### 🟢 Transacción LEGÍTIMA #1 — Compra normal con 3DS

```bash
curl -X POST http://127.0.0.1:8000/api/transacciones/ \
  -H "Content-Type: application/json" \
  -d '{
    "importe": "89.90",
    "card_brand": "visa",
    "card_type": "debito",
    "issuer_bank": "bcp",
    "payment_channel": "pago web",
    "eci_code": 5,
    "customer_region": "lima",
    "product_category": "repuestos_moto",
    "num_items": 2,
    "email_domain": "gmail.com",
    "bin": "404700"
  }'
```

**Resultado esperado:** Score bajo, **NO genera incidente**.

---

### 🟢 Transacción LEGÍTIMA #2 — Compra con Yape

```bash
curl -X POST http://127.0.0.1:8000/api/transacciones/ \
  -H "Content-Type: application/json" \
  -d '{
    "importe": "45.50",
    "card_brand": "visa",
    "card_type": "debito",
    "issuer_bank": "interbank",
    "payment_channel": "pago movil",
    "eci_code": 5,
    "customer_region": "arequipa",
    "product_category": "aceites_lubricantes",
    "num_items": 1,
    "email_domain": "hotmail.com",
    "bin": "454620",
    "wallet_yape": "si"
  }'
```

---

### 🟢 Transacción LEGÍTIMA #3 — Compra en cuotas Mastercard

```bash
curl -X POST http://127.0.0.1:8000/api/transacciones/ \
  -H "Content-Type: application/json" \
  -d '{
    "importe": "299.00",
    "card_brand": "mastercard",
    "card_type": "credito",
    "issuer_bank": "bbva",
    "payment_channel": "pago web",
    "eci_code": 2,
    "num_installments": 6,
    "customer_region": "cusco",
    "product_category": "cascos",
    "num_items": 1,
    "email_domain": "outlook.com",
    "bin": "520000"
  }'
```

---

### 🔴 Transacción FRAUDULENTA #1 — Email temporal, sin 3DS, monto alto

```bash
curl -X POST http://127.0.0.1:8000/api/transacciones/ \
  -H "Content-Type: application/json" \
  -d '{
    "importe": "2850.00",
    "card_brand": "visa",
    "card_type": "credito",
    "issuer_bank": "bcp",
    "payment_channel": "pago web",
    "eci_code": 0,
    "customer_region": "lima",
    "product_category": "electronica",
    "num_items": 1,
    "email_domain": "yopmail.com",
    "bin": "400000"
  }'
```

**Resultado esperado:** Score alto (>70%), **SÍ genera incidente automáticamente**.

---

### 🔴 Transacción FRAUDULENTA #2 — Card testing automatizado

```bash
curl -X POST http://127.0.0.1:8000/api/transacciones/ \
  -H "Content-Type: application/json" \
  -d '{
    "importe": "1599.99",
    "card_brand": "mastercard",
    "card_type": "credito",
    "issuer_bank": "scotiabank",
    "payment_channel": "app",
    "eci_code": 7,
    "customer_region": "piura",
    "product_category": "accesorios",
    "num_items": 1,
    "email_domain": "guerrillamail.com",
    "bin": "999999"
  }'
```

---

### 🔴 Transacción FRAUDULENTA #3 — Amex sospechosa

```bash
curl -X POST http://127.0.0.1:8000/api/transacciones/ \
  -H "Content-Type: application/json" \
  -d '{
    "importe": "3200.00",
    "card_brand": "amex",
    "card_type": "credito",
    "issuer_bank": "falabella",
    "payment_channel": "pago movil",
    "eci_code": 6,
    "customer_region": "callao",
    "product_category": "otros",
    "num_items": 1,
    "email_domain": "tempmail.com",
    "bin": "411111"
  }'
```


## 2. Simulación temporal (POST) — **No persiste en BD**

```
POST /api/simulacion-api/
Content-Type: application/json
```

Mismos payloads de arriba. La transacción **no se guarda**. Útil para pruebas rápidas.

---

## 3. Consultar Incidentes (GET)

```bash
# Todos los incidentes
curl http://127.0.0.1:8000/api/incidentes/

# Solo pendientes
curl "http://127.0.0.1:8000/api/incidentes/?estado=Pendiente"

# Ordenados por score desc
curl "http://127.0.0.1:8000/api/incidentes/?ordering=-score_riesgo"
```

---

## 4. Gestionar Incidente (PATCH)

```bash
# Confirmar como fraude (reemplazar {ID} con el ID real)
curl -X PATCH http://127.0.0.1:8000/api/incidentes/{ID}/cambiar_estado/ \
  -H "Content-Type: application/json" \
  -d '{"estado": "Fraude confirmado", "comentario": "Verificado: email temporal y patrón sospechoso"}'

# Marcar como falso positivo
curl -X PATCH http://127.0.0.1:8000/api/incidentes/{ID}/cambiar_estado/ \
  -H "Content-Type: application/json" \
  -d '{"estado": "Falso positivo", "comentario": "Cliente verificado por teléfono"}'
```

---

## 5. Consultar Transacciones (GET)

```bash
# Todas
curl http://127.0.0.1:8000/api/transacciones/

# Filtrar por marca
curl "http://127.0.0.1:8000/api/transacciones/?card_brand=visa"

# Filtrar por categoría
curl "http://127.0.0.1:8000/api/transacciones/?product_category=electronica"
```

---

## 📊 Flujo recomendado para el Video

1. **Mostrar Dashboard** → Datos históricos con gráficos
2. **Simulación Individual** → Botón "Caso Legítimo" → Simular → score bajo
3. **Simulación Individual** → Botón "Caso Fraude" → Simular → score alto + SHAP
4. **Terminal: POST transacción legítima** (curl #1) → score bajo, sin incidente
5. **Terminal: POST transacción fraudulenta** (curl #4) → score alto, incidente creado
6. **Pantalla Incidentes** → Ver el nuevo incidente (aparece en tiempo real)
7. **Detalle incidente** → Ver explicabilidad SHAP, factores de riesgo
8. **Gestionar** → Confirmar como fraude
9. **Dashboard** → Mostrar actualización del dinero protegido
10. **Terminal: POST otra fraudulenta** (curl #5) → confirmar detección consistente

---

## 📦 Campos del API

### Requeridos (datos del comercio)
| Campo | Tipo | Ejemplo |
|-------|------|---------|
| `importe` | decimal | `"89.90"` |
| `card_brand` | string | `visa`, `mastercard`, `amex`, `diners` |
| `card_type` | string | `credito`, `debito` |
| `issuer_bank` | string | `bcp`, `bbva`, `interbank`, `scotiabank` |
| `customer_region` | string | `lima`, `arequipa`, `piura`, `cusco` |
| `product_category` | string | `electronica`, `repuestos_moto`, `cascos` |

### Opcionales (el sistema asigna defaults)
| Campo | Default | Descripción |
|-------|---------|-------------|
| `payment_channel` | `pago web` | Canal: pago web, pago movil, app |
| `eci_code` | `5` | ECI autenticación 3DS |
| `num_items` | `1` | Cantidad de ítems |
| `num_installments` | `0` | Cuotas (0=contado) |
| `email_domain` | `gmail.com` | Dominio email cliente |
| `bin` | `404700` | BIN tarjeta (6 dígitos) |
| `wallet_yape` | `no` | Usa Yape: si/no |
| `wallet_plin` | `no` | Usa Plin: si/no |

### Calculados por el sistema (incluidos en la respuesta)

Estos ratios se calculan automáticamente usando el historial de transacciones en la BD (Algorithm 2).
**No se aceptan como input** — el sistema siempre los calcula y los guarda en la transacción.

| Ratio | Cálculo | Significado |
|-------|---------|-------------|
| `ratio_aar` | `monto / promedio_BIN` | Anomalía de monto vs histórico |
| `ratio_cmr` | `monto / mediana_categoría` | Anomalía de monto vs categoría |
| `ratio_asi` | `% tx con 3DS del BIN` | Fortaleza de autenticación |
| `ratio_vrr` | `tx_24h / promedio_diario_BIN` | Velocidad transaccional |
| `ratio_dar` | `denegaciones_24h / intentos_24h` | Tasa de denegación |
| `ratio_csi` | `regiones_distintas_por_BIN` | Índice de compartición |
| `ratio_dpe` | `entropía(razones_denegación)` | Diversidad de rechazos |

### Internos (NO enviar)
`transaction_status`, `action_code` y `denial_reason` son asignados automáticamente.

---

## 📨 Ejemplo de Respuesta del API

Al crear una transacción, la respuesta incluye los ratios calculados:

```json
{
  "id_transaccion": 46,
  "importe": "2850.00",
  "card_brand": "visa",
  "customer_region": "lima",
  "ratio_aar": "12.3987",
  "ratio_cmr": "1.6025",
  "ratio_asi": "0.8000",
  "ratio_vrr": "6.7500",
  "ratio_dar": "0.0000",
  "ratio_csi": "4.0000",
  "ratio_dpe": "0.0000",
  "score_riesgo": 82.35,
  "incidente_generado": true,
  "explicabilidad": { "..." }
}
```
