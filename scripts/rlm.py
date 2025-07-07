# 1. Importar librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# 2. Cargar los datos
df = pd.read_excel("jugadores_codificados.xlsx")

# 3. Identificar columnas de la variable objetivo (one-hot)
pos_cols = [col for col in df.columns if col.startswith('Posición_')]

# 4. Definir X (atributos) e y (clase)
X = df.drop(columns=['№', 'Jugadores'] + pos_cols)
y = df[pos_cols].idxmax(axis=1)  # Convertimos a una sola clase (ej. Posición_portero)

# 5. Normalizar los datos (importante para regresión logística)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Entrenar modelo de Regresión Logística Multiclase
modelo_log = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
modelo_log.fit(X_train, y_train)

# 8. Realizar predicciones
y_pred = modelo_log.predict(X_test)

# 9. Evaluar el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score macro:", f1_score(y_test, y_pred, average='macro'))
print("\nReporte completo:\n", classification_report(y_test, y_pred))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
