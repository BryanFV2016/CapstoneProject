import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Cargar datos
df = pd.read_excel("jugadores_codificados.xlsx")

# Variables predictoras y objetivo
X = df.drop(columns=['№', 'Jugadores', 'Posición_delantero', 'Posición_mediocampista',
                     'Posición_defensa', 'Posición_portero'], errors='ignore')
y = df[['Posición_delantero', 'Posición_mediocampista', 'Posición_defensa', 'Posición_portero']].idxmax(axis=1)

# Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Etiquetas para mostrar
etiquetas = le.classes_

from sklearn.linear_model import LogisticRegression

modelo_lr = LogisticRegression(max_iter=1000)
modelo_lr.fit(X_train, y_train)
y_pred_lr = modelo_lr.predict(X_test)

cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=etiquetas, yticklabels=etiquetas)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Prediction")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

