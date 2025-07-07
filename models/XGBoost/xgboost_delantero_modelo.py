import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# --- Cargar el dataset original ---
df = pd.read_excel("jugadores_codificados.xlsx")

# --- Identificar columnas de la variable objetivo ---
pos_cols = [col for col in df.columns if col.startswith('Posición_')]

# --- Crear variables derivadas que ayuden al modelo ---
df['Indice_Tecnico'] = (df['Precisión Pase (%)'] + df['Técnica Disparo']) / 2
df['Velocidad_IMC'] = df['Velocidad Sprint (km/h)'] / df['IMC']
df['Densidad_Corporal'] = df['Peso'] / (df['Altura'] ** 2)

# --- Separar X e y ---
X = df.drop(columns=['№', 'Jugadores'] + pos_cols)  # Variables predictoras
y_text = df[pos_cols].idxmax(axis=1)

# --- Codificar etiquetas a números ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_text)

# --- Normalizar características ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Aplicar SMOTE para balancear clases ---
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

# --- Dividir en entrenamiento y prueba ---
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# --- Entrenar modelo XGBoost ---
modelo_xgb = XGBClassifier(
    max_depth=4,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.9,
    colsample_bytree=0.9,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
modelo_xgb.fit(X_train, y_train)

# --- Realizar predicciones ---
y_pred = modelo_xgb.predict(X_test)

# --- Decodificar etiquetas para interpretación ---
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# --- Evaluar rendimiento del modelo ---
print("=== Resultados Finales de XGBoost ===")
print("Accuracy:", accuracy_score(y_test_labels, y_pred_labels))
print("F1 Score macro:", f1_score(y_test_labels, y_pred_labels, average='macro'))
print("\nReporte de Clasificación:\n", classification_report(y_test_labels, y_pred_labels))
print("Matriz de Confusión:\n", confusion_matrix(y_test_labels, y_pred_labels))

# --- Visualizar matriz de confusión ---
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Prediction")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
