import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Cargar datos
df = pd.read_excel("jugadores_codificados.xlsx")

# Preparar variables
X = df.drop(columns=["№", "Jugadores", "Posición_delantero", "Posición_mediocampista", "Posición_defensa", "Posición_portero"], errors="ignore")
y = df[["Posición_delantero", "Posición_mediocampista", "Posición_defensa", "Posición_portero"]].idxmax(axis=1)

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# División
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Entrenamiento modelo final (XGBoost)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", objective="multi:softprob", num_class=4)
model.fit(X_train, y_train)

# Seleccionar un jugador para predecir su posición
jugador = X_test.iloc[[0]]  # Puedes cambiar el índice
prediccion = model.predict(jugador)[0]
nombre_clase = label_encoder.inverse_transform([prediccion])[0]

print("📌 Predicción de posición para el jugador:")
print(jugador)
print("\n🔎 Posición estimada:", nombre_clase)
