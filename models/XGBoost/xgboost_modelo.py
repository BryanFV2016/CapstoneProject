import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# --- Cargar archivo Excel ---
df = pd.read_excel("jugadores_codificados.xlsx")  # asegúrate de que esté en la misma carpeta

# --- Detectar columnas de posición (one-hot) ---
pos_cols = [col for col in df.columns if col.startswith('Posición_')]

# --- Separar características (X) y etiquetas (y) ---
X = df.drop(columns=['№', 'Jugadores'] + pos_cols)
y_text = df[pos_cols].idxmax(axis=1)

# --- Codificar y en números con LabelEncoder ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_text)  # ahora y son enteros: 0, 1, 2, 3

# --- Normalizar las características ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Aplicar SMOTE para balancear las clases ---
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

# --- Dividir en entrenamiento y prueba ---
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# --- Entrenar modelo XGBoost ---
modelo_xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
modelo_xgb.fit(X_train, y_train)

# --- Realizar predicciones ---
y_pred = modelo_xgb.predict(X_test)

# --- Decodificar etiquetas numéricas a texto ---
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# --- Evaluar el modelo ---
print("=== Resultados de XGBoost ===")
print("Accuracy:", accuracy_score(y_test_labels, y_pred_labels))
print("F1 Score macro:", f1_score(y_test_labels, y_pred_labels, average='macro'))
print("\nReporte de Clasificación:\n", classification_report(y_test_labels, y_pred_labels))
print("Matriz de Confusión:\n", confusion_matrix(y_test_labels, y_pred_labels))
