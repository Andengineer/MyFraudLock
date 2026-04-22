import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "outputs" / "models"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_metrics():
    path = MODEL_DIR / "comparison_metrics.json"
    if not path.exists():
        print("❌ No se encontró comparison_metrics.json. Ejecuta 04_train_optimized.py primero.")
        return None
    with open(path, "r") as f:
        return json.load(f)

def generate_plots():
    data = load_metrics()
    if not data:
        return
    
    print("✓ Archivo de métricas encontrado. Regenerando gráficas F1...")
    
    # 1. Bar plot comparativo (F1 Score)
    default = data.get("default", {})
    optimized = data.get("optimized", {})
    
    models = list(default.keys())
    if not models:
        return
        
    f1_default = [default[m]["metrics"].get("f1", 0) for m in models]
    f1_opt = [optimized[m]["metrics"].get("f1", 0) for m in models]
    
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, f1_default, width, label='Default', color='#A9CCE3')
    plt.bar(x + width/2, f1_opt, width, label='Optimizado (F1 Sweeping)', color='#2471A3')
    
    plt.ylabel('F1-Score')
    plt.title('Comparación de Equilibrio (F1-Score) por Modelo')
    plt.xticks(x, models)
    plt.legend()
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_f1_score_comparasion.png", dpi=300)
    plt.close()
    
    print("  ✓ Gráfica 01_f1_score_comparasion.png generada.")
    print("Todo listo. Ahora puedes alterar este archivo libremente para modificar la tesis.")

if __name__ == "__main__":
    generate_plots()
