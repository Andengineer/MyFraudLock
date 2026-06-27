# Diagrama de Clases y Módulos (MyFraudLock)

Puede copiar e incrustar este código en sus documentos de tesis compatibles con Markdown o visores visuales (como Obsidian, GitHub o Mermaid Live Editor) para generar un gráfico UML:

```mermaid
classDiagram
    class DjangoServer {
        +runserver()
    }
    
    class TransaccionesAPI {
        +recibir_transaccion(JSON)
        +validar_datos()
        +enviar_a_ml()
    }
    
    class ModelLayerXAI {
        -tf_model: Model
        -shap_explainer: DeepExplainer
        +predict_fraud(tx: dict) : tuple
        +_ensure_artifacts()
        +predict_and_explain(payload: dict) : json
    }
    
    class DBHandler {
        +guardar_transaccion()
        +generar_incidente()
    }
    
    class WebViews {
        +dashboard_view(request)
        +incidentes_listado(request)
        +incidente_detalle_view(request)
        +simulacion_lote_view(request)
    }

    class Frontend {
        +graficos_financieros(ChartJS)
        +grafico_shapley(Javascript HTML)
        +long_polling_nuevos_incidentes()
    }

    DjangoServer --> TransaccionesAPI : "Rutas (urls.py)"
    DjangoServer --> WebViews : "Vistas Autenticadas"
    TransaccionesAPI --> ModelLayerXAI : "Llama inferencia"
    ModelLayerXAI --> DBHandler : "Devuelve score y SHAP"
    WebViews --> Frontend : "Renderiza Contexto"
    Frontend --> TransaccionesAPI : "Envía simulaciones"
```
