import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ========== Crear carpeta para resultados ==========
carpeta_resultados = "resultados_comparacion"
os.makedirs(carpeta_resultados, exist_ok=True)

# ========== Cargar métricas desde archivos CSV ==========

rutas_csv = {
    "YOLOv11": "evaluate_yolov11/metrics.csv",
    "YOLOv8": "evaluate_yolov8/metrics.csv",
    "Faster R-CNN": "evaluate_fasterrcnn/metrics.csv",
    "EfficientDet": "evaluate_efficientdet/metrics.csv"
}

# Lista para almacenar métricas
data = []

for modelo, ruta in rutas_csv.items():
    df = pd.read_csv(ruta)
    mAP50 = float(df.loc[0, "mAP@0.5 (%)"])
    precision = float(df.loc[0, "Precision (%)"])
    recall = float(df.loc[0, "Recall (%)"])
    f1 = float(df.loc[0, "F1 Score (%)"])
    data.append([modelo, mAP50, precision, recall, f1])

# Crear DataFrame comparativo
df_comparacion = pd.DataFrame(data, columns=["Modelo", "mAP@0.5 (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"])
print(df_comparacion)

# Guardar el DataFrame como CSV
csv_path = os.path.join(carpeta_resultados, "tabla_comparativa.csv")
df_comparacion.to_csv(csv_path, index=False)
print(f"Tabla guardada en: {csv_path}")

# ========== Gráfico de barras agrupadas ==========

plt.figure(figsize=(10, 6))
df_melted = df_comparacion.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')
sns.barplot(data=df_melted, x='Métrica', y='Valor', hue='Modelo')
plt.title("Comparación de métricas por modelo")
plt.ylim(0, 110)
plt.ylabel("Porcentaje (%)")
plt.legend(title='Modelo')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Guardar el gráfico
grafico_path = os.path.join(carpeta_resultados, "grafico_comparativo.png")
plt.savefig(grafico_path, dpi=300)
print(f"Gráfico guardado en: {grafico_path}")

plt.show()
