# Diagrama de Base de Datos (MyFraudLock)

Puede copiar e incrustar este código en sus documentos de tesis compatibles con Markdown o visores visuales (como Obsidian, GitHub o Mermaid Live Editor) para generar un gráfico ER impecable:

```mermaid
erDiagram
    Usuario {
        int id_usuario PK
        string username
        string email
        string password
        string telefono
        boolean activo
        string rol "ADMIN, ANALISTA, EJECUTIVO"
    }

    Transaccion {
        int id_transaccion PK
        decimal importe
        datetime fecha
        string card_brand
        string card_type
        string issuer_bank
        string payment_channel
        int eci_code
        int num_installments
        string customer_region
        int city_population
        boolean is_new_customer
        int days_since_first_purchase
        decimal avg_historical_amount
        string category
        int num_items
        boolean has_discount
        int previous_failed_attempts
    }

    Incidente {
        int id_incidente PK
        int gestionado_por_id FK "Refs Usuario"
        int id_transaccion_id FK "Refs Transaccion"
        string comentario
        string estado "Pendiente, Fraude confirmado, Falso positivo"
        boolean es_fraude
        datetime fecha
        decimal score_riesgo
        json explicabilidad "SHAP Values"
    }

    Configuracion {
        int id PK
        int umbral_score "ej. 70"
        boolean notificaciones_email
        datetime actualizado_en
        int actualizado_por_id FK "Refs Usuario"
    }

    %% Relaciones
    Usuario ||--o{ Incidente : "Gestiona (1 a muchos)"
    Transaccion ||--o| Incidente : "Genera (1 a 1)"
    Usuario ||--o{ Configuracion : "Actualiza (1 a muchos)"
```
