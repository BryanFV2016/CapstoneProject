# ---------------------------------------------------------------
# Visualización de resultados del modelo final (XGBoost + binario)
# Genera: matriz de confusión, F1 por clase, comparación de modelos
# ---------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Etiquetas verdaderas (decodificadas) y predichas (modelo final)
# Reemplaza estas listas con tus datos reales si lo haces por separado
y_test_labels = [
    'Posición_defensa', 'Posición_delantero', 'Posición_mediocampista', 'Posición_portero',
    # ... continuar con todos los reales
]
y_pred_final = [
    'Posición_defensa', 'Posición_delantero', 'Posición_mediocampista', 'Posición_portero',
    # ... continuar con todas las predicciones
]

# ---------------------------------------------------------------
# 1. Matriz de Confusión
# ---------------------------------------------------------------
labels = ['Posición_defensa', 'Posición_delantero', 'Posición_mediocampista', 'Posición_portero']
cm = confusion_matrix(y_test_labels, y_pred_final, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Matriz de Confusión - Modelo XGBoost Final')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 2. F1-Score por Clase
# ---------------------------------------------------------------
report = classification_report(y_test_labels, y_pred_final, target_names=labels, output_dict=True)
f1_scores = [report[label]['f1-score'] for label in labels]

plt.figure(figsize=(8, 5))
sns.barplot(x=labels, y=f1_scores, palette='viridis')
plt.ylim(0, 1)
plt.title('F1-Score por Clase - Modelo XGBoost Final')
plt.ylabel('F1-Score')
plt.xlabel('Clase')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 3. Comparación de Accuracy entre Modelos Evaluados
# ---------------------------------------------------------------
modelos = ['Reg. Logística', 'Random Forest', 'MLP', 'XGBoost Base', 'XGB Mejorado', 'XGB Final']
accuracies = [0.528, 0.593, 0.636, 0.697, 0.712, 0.773]

plt.figure(figsize=(9, 5))
sns.barplot(x=modelos, y=accuracies, palette='Set2')
plt.ylim(0.4, 0.85)
plt.title('Accuracy comparison between models')
plt.ylabel('Accuracy')
plt.xlabel('Modelo')
plt.tight_layout()
plt.show()
