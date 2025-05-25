# 1. Importar librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# 2. Cargar los datos ya codificados
df = pd.read_excel("jugadores_codificados.xlsx")

# 3. Definir variables predictoras (X) y variable objetivo (y)
pos_cols = [col for col in df.columns if col.startswith('Posición_')]  # identificar columnas objetivo
X = df.drop(columns=['№', 'Jugadores'] + pos_cols)  # eliminar columnas no numéricas ni objetivo
y = df[pos_cols].idxmax(axis=1)  # extraer la clase (posición) con valor 1

# 4. Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entrenar modelo Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# 6. Hacer predicciones
y_pred = modelo_rf.predict(X_test)

# 7. Evaluar el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision macro:", precision_score(y_test, y_pred, average='macro'))
print("Recall macro:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score macro:", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
