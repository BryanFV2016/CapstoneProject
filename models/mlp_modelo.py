import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# --- Cargar el archivo Excel ---
df = pd.read_excel("jugadores_codificados.xlsx")  # asegúrate de que esté en la misma carpeta

# --- Detectar columnas de clase (one-hot) ---
pos_cols = [col for col in df.columns if col.startswith('Posición_')]

# --- Separar características (X) y etiqueta (y) ---
X = df.drop(columns=['№', 'Jugadores'] + pos_cols)
y = df[pos_cols].idxmax(axis=1)  # devuelve por ejemplo: "Posición_portero"

# --- Normalizar datos ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Aplicar SMOTE para balancear las clases ---
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# --- Dividir en conjunto de entrenamiento y prueba ---
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# --- Definir y entrenar el modelo MLP ---
modelo_mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
modelo_mlp.fit(X_train, y_train)

# --- Realizar predicciones ---
y_pred = modelo_mlp.predict(X_test)

# --- Evaluar el modelo ---
print("=== Resultados del MLP ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score macro:", f1_score(y_test, y_pred, average='macro'))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
