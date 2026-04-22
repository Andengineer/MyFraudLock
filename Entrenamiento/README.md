# 🧠 Pipeline de ML & Entrenamiento (MyFraudLock)

Este directorio contiene todo el ecosistema de generación de datos sintéticos, Análisis Exploratorio (EDA), Experimentación de Balanceo y Entrenamiento Optimizado de modelos de Inteligencia Artificial desplegados en la aplicación principal de **MyFraudLock**.

## 📌 Requisitos Previos

```bash
pip install pandas numpy scikit-learn tensorflow keras matplotlib seaborn shap joblib imbalanced-learn xgboost optuna
```

## 📂 Pipeline de Scripts

El pipeline científico se ejecuta en **5 fases secuenciales**:

### 1) `01_generate_data.py` — Generación de Dataset Sintético
Genera **50,000 transacciones** con distribuciones estadísticas extraídas de datos reales de e-commerce peruano (Niubiz + WooCommerce). Implementa 6 tipologías de fraude: Card Testing, Stolen Card, Account Takeover, Friendly Fraud, Bot Automated y Triangulation.

**Salida:** `outputs/data/fraud_ecommerce_dataset.csv`

### 2) `02_eda_analysis.py` — Análisis Exploratorio de Datos
Genera 10 gráficos de alta calidad para tesis/paper: distribución de clases, montos, patrones temporales, análisis categórico, tipos de fraude, correlación, análisis ECI, features comportamentales, análisis geográfico y resumen estadístico.

**Salida:** `outputs/eda/*.png`

### 3) `03_balance_experiment.py` — Fase 1: Experimentación de Balanceo
Compara **3 técnicas de balanceo × 5 modelos = 15 combinaciones**:

| Técnica | Estrategia |
|:--------|:-----------|
| Sin Balanceo | Solo class_weight automático |
| SMOTE-Tomek | Oversampling + limpieza de frontera |
| ADASYN | Oversampling adaptativo en zonas difíciles |

| Modelo | Tipo |
|:-------|:-----|
| DNN | Deep Neural Network (Feed-Forward) |
| CNN-1D | Convolutional Neural Network 1D |
| RNN-GRU | Recurrent Neural Network (Gated Recurrent Unit) |
| Autoencoder | Autoencoder + Classification Head |
| XGBoost | Gradient Boosting (ML clásico) |

**Métricas:** AUC-ROC, AUC-PR, F1-Score, Precisión, Recall, FPR, MCC, G-Mean, Costo de Negocio (FN×10 + FP×1)

**Salida:** `outputs/figures/01-06_balance_*.png`, `outputs/models/balance_experiment_results.json`

### 4) `04_train_optimized.py` — Fase 2: Optimización de Hiperparámetros
Toma el mejor balanceo de la Fase 1 y ejecuta:
1. Entrenamiento con **hiperparámetros default** (5 modelos)
2. Optimización con **Optuna** (20 trials bayesianos por modelo)
3. Comparativa final **Default vs Optimizado**
4. Si XGBoost gana, se documenta pero se exporta el mejor modelo DL
5. Análisis SHAP del modelo ganador

**Salida:** `outputs/figures/07-11_*.png`, `outputs/models/best_model.keras`, `outputs/explainability/*.png`

### 5) `05_export_model.py` — Exportación al Backend Django
Copia los artefactos de producción al directorio `api/ml/`:
- `best_model.keras` — Pesos neuronales
- `preprocessor.joblib` — Pipeline de Scale y One-Hot
- `feature_names.json` — Nombres internos de features
- `group_map.json` — Agrupación de features para XAI
- `background.npy` — 200 filas de fondo para SHAP values

## 🚀 ¿Cómo reproducir?

```bash
# Desde este directorio:
python 01_generate_data.py        # ~2 seg
python 02_eda_analysis.py         # ~10 seg
python 03_balance_experiment.py   # ~30-45 min (15 entrenamientos)
python 04_train_optimized.py      # ~20-30 min (Optuna + reentrenamiento)
python 05_export_model.py         # ~1 seg (copia artefactos)
```

> **Nota:** El script `05_export_model.py` copia automáticamente los resultados al directorio `../api/ml/`, por lo que el Backend de Django los leerá inmediatamente.

## 📊 Estructura de Salidas

```
outputs/
├── data/               # Dataset generado y metadata
├── eda/                # 10 gráficos de análisis exploratorio
├── figures/            # Gráficos comparativos (Fase 1 y 2)
├── models/             # Modelos, preprocesador, métricas
└── explainability/     # SHAP importance y summary plots
```
