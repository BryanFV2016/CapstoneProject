import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar datos
df = pd.read_excel("jugadores_codificados.xlsx")

# 2. Separar variables independientes y dependiente
X = df.drop(columns=['№', 'Jugadores', 'Posición_delantero', 'Posición_mediocampista', 'Posición_defensa', 'Posición_portero'], errors='ignore')
y = df[['Posición_delantero', 'Posición_mediocampista', 'Posición_defensa', 'Posición_portero']].idxmax(axis=1)

# 3. Codificar variable objetivo
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. Entrenar modelo MLP con scikit-learn
model_mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
model_mlp.fit(X_train, y_train)

# 6. Predicción y matriz de confusión
y_pred = model_mlp.predict(X_test)
matriz_mlp = confusion_matrix(y_test, y_pred)

# 7. Graficar la matriz de confusión
labels = le.classes_
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_mlp, annot=True, cmap="Blues", fmt="d", 
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - MLP")
plt.xlabel("Prediction")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
