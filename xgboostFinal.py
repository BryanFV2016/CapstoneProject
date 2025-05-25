import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# --- Cargar archivo Excel ---
df = pd.read_excel("jugadores_codificados.xlsx")

# --- Variables derivadas ofensivas ---
df['Indice_Tecnico'] = (df['Precisión Pase (%)'] + df['Técnica Disparo']) / 2
df['Velocidad_IMC'] = df['Velocidad Sprint (km/h)'] / df['IMC']
df['Densidad_Corporal'] = df['Peso'] / (df['Altura'] ** 2)
df['Agresividad_Ofensiva'] = df['Técnica Disparo'] * df['Velocidad Sprint (km/h)'] / df['Peso']
df['Potencia_Relativa'] = df['Técnica Disparo'] / df['Altura']

# --- Preparar X e y ---
pos_cols = [col for col in df.columns if col.startswith('Posición_')]
X = df.drop(columns=['№', 'Jugadores'] + pos_cols)
y_text = df[pos_cols].idxmax(axis=1)

# --- Codificación de y ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_text)
delantero_label = 'Posición_delantero'
delantero_index = np.where(label_encoder.classes_ == delantero_label)[0][0]

# --- Normalizar ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- SMOTE multiclase ---
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y_encoded)

# --- División train-test ---
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# --- Ajuste de pesos de clase ---
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
peso_delantero = class_weights[delantero_index]

# --- Modelo multiclase con peso para delantero ---
modelo_multi = XGBClassifier(
    scale_pos_weight=peso_delantero,
    max_depth=4,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.9,
    colsample_bytree=0.9,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
modelo_multi.fit(X_train, y_train)

# --- Modelo binario auxiliar: delantero vs no-delantero ---
y_bin = (y_encoded == delantero_index).astype(int)
Xb_res, yb_res = smote.fit_resample(X_scaled, y_bin)
Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb_res, yb_res, test_size=0.2, random_state=42)

modelo_binario = XGBClassifier(
    max_depth=4,
    learning_rate=0.1,
    n_estimators=150,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
modelo_binario.fit(Xb_train, yb_train)

# --- Predicción multiclase con ajuste de umbral para delantero ---
proba_multi = modelo_multi.predict_proba(X_test)
proba_bin = modelo_binario.predict_proba(X_test)
y_pred_final = []

for i in range(len(X_test)):
    proba_delantero = proba_multi[i][delantero_index]
    pred_idx = np.argmax(proba_multi[i])
    pred_clase = label_encoder.classes_[pred_idx]

    # Si no predice delantero, pero binario lo afirma con fuerza
    if pred_clase != 'Posición_delantero' and proba_bin[i][1] >= 0.75:
        y_pred_final.append('Posición_delantero')
    else:
        y_pred_final.append(pred_clase)

# --- Decodificar etiquetas reales ---
y_test_labels = label_encoder.inverse_transform(y_test)

# --- Evaluación final ---
print("=== RESULTADOS MEJORADOS PARA DELANTEROS ===")
print("Accuracy:", accuracy_score(y_test_labels,  y_pred_final))
print("F1 Score macro:", f1_score(y_test_labels,  y_pred_final, average='macro'))
print("\nClassification Report:\n", classification_report(y_test_labels,  y_pred_final))
print("Matriz de Confusión:\n", confusion_matrix(y_test_labels,  y_pred_final))

# --- Visualizar matriz de confusión ---
cm = confusion_matrix(y_test_labels,  y_pred_final, labels=label_encoder.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix - Improved XGBoost")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
