import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Diccionario con los resultados reales
resultados_modelos = {
    "Modelo": ["Logistic Regression", "Random Forest", "MLP", "XGBoost Base", "XGBoost Mejorado",  "XGBoost Final"],
    "Accuracy": [0.5278, 0.6364, 0.6364, 0.6980, 0.7122, 0.7727],
    "Precision": [0.53, 0.61, 0.63, 0.69, 0.71, 0.76],
    "Recall": [0.55, 0.59, 0.64, 0.58, 0.63, 0.77],
    "F1 Score": [0.54, 0.60, 0.62, 0.57, 0.60, 0.76]
}

# Crear DataFrame
df = pd.DataFrame(resultados_modelos)

# Configuración del gráfico
etiquetas = df["Modelo"]
x = np.arange(len(etiquetas))  # posiciones X
ancho = 0.2  # ancho de cada barra

fig, ax = plt.subplots(figsize=(12, 6))

# Crear barras para cada métrica
b1 = ax.bar(x - 1.5*ancho, df["Accuracy"], width=ancho, label="Accuracy")
b2 = ax.bar(x - 0.5*ancho, df["Precision"], width=ancho, label="Precision")
b3 = ax.bar(x + 0.5*ancho, df["Recall"], width=ancho, label="Recall")
b4 = ax.bar(x + 1.5*ancho, df["F1 Score"], width=ancho, label="F1 Score")

# Etiquetas en cada barra
def agregar_etiquetas(barras):
    for barra in barras:
        height = barra.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(barra.get_x() + barra.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

for barras in [b1, b2, b3, b4]:
    agregar_etiquetas(barras)

# Título y ejes
ax.set_title("Performance Metrics by Model")
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(etiquetas, rotation=30, ha='right')
ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()
plt.show()
