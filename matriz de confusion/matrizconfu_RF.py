import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar datos
df = pd.read_excel("jugadores_codificados.xlsx")

# 2. Definir X (features) y y (clase)
X = df.drop(columns=['№', 'Jugadores', 'Posición_delantero', 'Posición_mediocampista', 'Posición_defensa', 'Posición_portero'], errors='ignore')
y = df[['Posición_delantero', 'Posición_mediocampista', 'Posición_defensa', 'Posición_portero']].idxmax(axis=1)

# 3. Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Separar en entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. Entrenar modelo Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# 6. Predecir y obtener matriz de confusión
y_pred = model_rf.predict(X_test)
matriz_rf = confusion_matrix(y_test, y_pred)

# 7. Etiquetas de clase (en el orden del label encoder)
labels = le.classes_

# 8. Graficar matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_rf, annot=True, cmap="Blues", fmt="d",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Prediction")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
