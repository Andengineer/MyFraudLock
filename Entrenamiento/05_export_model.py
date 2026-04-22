#!/usr/bin/env python3
"""
05_export_model.py — Exporta el mejor modelo y artefactos para MyFraudLock.
Copia el modelo entrenado, preprocesador, feature names, group map y background
al directorio api/ml/ del proyecto Django.

NOTA: Solo se exportan modelos Keras (.keras). Si el ganador global fue
XGBoost, la Fase 2 ya se encargó de exportar el mejor modelo DL como
best_model.keras.
"""

import shutil, json
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
MODEL_DIR  = BASE_DIR / "outputs" / "models"
PROJECT_ML = BASE_DIR.parent / "api" / "ml"

def export():
    print("=" * 60)
    print("EXPORTACIÓN DE MODELO A MyFraudLock")
    print("=" * 60)

    PROJECT_ML.mkdir(parents=True, exist_ok=True)

    files = {
        "best_model.keras":    "best_model.keras",
        "preprocessor.joblib": "preprocessor.joblib",
        "feature_names.json":  "feature_names.json",
        "group_map.json":      "group_map.json",
        "background.npy":      "background.npy",
    }

    for src_name, dst_name in files.items():
        src = MODEL_DIR / src_name
        dst = PROJECT_ML / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ {src_name} → {dst}")
        else:
            print(f"  ⚠ {src_name} no encontrado")

    # Nota: Los JSON analíticos como comparison_metrics.json se mantienen
    # de forma aislada e independiente en Entrenamiento/outputs/models/
    # y ya NO se exportan al Backend por orden arquitectónica.

    # Read and display summary
    metrics_path = MODEL_DIR / "comparison_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        # New format from Phase 2
        if "export_model" in metrics:
            export_name = metrics["export_model"]
            overall = metrics.get("overall_winner", export_name)
            opt = metrics.get("optimized", {})
            if export_name in opt:
                best = opt[export_name]
                print(f"\n  🏆 Ganador global: {metrics.get('overall_winner', '?')}")
                print(f"  📦 Modelo exportado: {best['label']}")
                print(f"     Balanceo:  {metrics.get('best_balance', '?')}")
                print(f"     AUC-ROC:   {best['metrics']['auc_roc']:.4f}")
                print(f"     AUC-PR:    {best['metrics']['auc_pr']:.4f}")
                print(f"     F1-Score:  {best['metrics']['f1']:.4f}")
                print(f"     Recall:    {best['metrics']['recall']:.4f}")
                print(f"     MCC:       {best['metrics']['mcc']:.4f}")
            if overall == "XGBoost":
                print(f"\n  ℹ️  XGBoost ganó la comparativa pero NO fue exportado.")
                print(f"      Se exportó el mejor modelo Deep Learning ({best['label']}).")
        else:
            # Legacy format
            best_name = max(metrics.keys(),
                            key=lambda k: metrics[k]["metrics"]["auc_roc"])
            best = metrics[best_name]
            print(f"\n  🏆 Modelo exportado: {best['label']}")
            print(f"     AUC-ROC:   {best['metrics']['auc_roc']:.4f}")

    print("\n✅ Exportación completada")
    print(f"   Destino: {PROJECT_ML}")

if __name__ == "__main__":
    export()
