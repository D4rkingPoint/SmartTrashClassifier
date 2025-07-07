import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar resultados YOLOv8 y YOLOv11
csv_yolo8 = pd.read_csv("runs/Entrenamiento_yolov8/train3/results.csv")
csv_yolo11 = pd.read_csv("runs/Entrenamiento_yolov11_new/train13/results.csv")

def extraer_metricas(row):
    mAP50 = round(row["metrics/mAP50(B)"] * 100, 2)
    precision = round(row["metrics/precision(B)"] * 100, 2)
    recall = round(row["metrics/recall(B)"] * 100, 2)
    f1 = round(2 * precision * recall / (precision + recall + 1e-8), 2)
    return mAP50, precision, recall, f1

# Última fila con resultados
last_yolo8 = csv_yolo8.iloc[-1]
last_yolo11 = csv_yolo11.iloc[-1]
yolo8_metrics = extraer_metricas(last_yolo8)
yolo11_metrics = extraer_metricas(last_yolo11)

# Cargar métricas Faster R-CNN y EfficientDet (archivos CSV generados por evaluate scripts)
df_faster = pd.read_csv("evaluate_fasterrcnn/metrics.csv")
df_efficientdet = pd.read_csv("evaluate_efficientdet/metrics.csv")

# Extraer métricas de Faster y Efficientdet
faster_metrics = (
    float(df_faster.loc[0, "mAP@0.5 (%)"]),
    float(df_faster.loc[0, "Precision (%)"]),
    float(df_faster.loc[0, "Recall (%)"]),
    float(df_faster.loc[0, "F1 Score (%)"])
)

efficientdet_metrics = (
    float(df_efficientdet.loc[0, "mAP@0.5 (%)"]),
    float(df_efficientdet.loc[0, "Precision (%)"]),
    float(df_efficientdet.loc[0, "Recall (%)"]),
    float(df_efficientdet.loc[0, "F1 Score (%)"])
)

# Crear DataFrame comparativo
df_comparacion = pd.DataFrame([
    ["YOLOv11", *yolo11_metrics],
    ["YOLOv8", *yolo8_metrics],
    ["Faster R-CNN", *faster_metrics],
    ["EfficientDet", *efficientdet_metrics]
], columns=["Modelo", "mAP@0.5 (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"])

print(df_comparacion)


# ----------- Gráfico de barras agrupadas -----------

plt.figure(figsize=(10,6))
df_melted = df_comparacion.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')
sns.barplot(data=df_melted, x='Métrica', y='Valor', hue='Modelo')
plt.title("Comparación de métricas por modelo")
plt.ylim(0, 110)
plt.ylabel("Porcentaje (%)")
plt.legend(title='Modelo')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ----------- Función para vértices poligonales -----------

def unit_poly_verts(theta):
    """Devuelve los vértices para el polígono del radar (normalizado en círculo unitario)"""
    x0, y0, r = [0.5] * 3
    verts = [(r * np.cos(t) + x0, r * np.sin(t) + y0) for t in theta]
    return verts

# ----------- Función radar_factory corregida -----------

def radar_factory(num_vars, frame='circle'):
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.spines import Spine
    from matplotlib.path import Path
    from matplotlib.projections import register_projection

    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                line.set_clip_on(False)
            return lines

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return plt.Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return plt.Polygon(unit_poly_verts(theta), closed=True, edgecolor='k')
            else:
                raise ValueError(f"Unknown frame: {frame}")

    register_projection(RadarAxes)
    return theta

# ----------- Gráfico de Radar -----------

metrics = ["mAP@0.5 (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"]
num_vars = len(metrics)
theta = radar_factory(num_vars, frame='polygon')

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='radar'))
ax.set_title('Radar de desempeño de modelos', weight='bold', size='medium', position=(0.5, 1.1),
             horizontalalignment='center', verticalalignment='center')

for i, row in df_comparacion.iterrows():
    values = row[metrics].values.flatten().tolist()
    values += values[:1]  # cerrar polígono
    theta_closed = np.append(theta, theta[0])  # cerrar ciclo en theta también
    ax.plot(theta_closed, values, label=row['Modelo'])
    ax.fill(theta_closed, values, alpha=0.25)


ax.set_varlabels(metrics)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.show()