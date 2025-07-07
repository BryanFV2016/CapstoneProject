import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Cargar tu archivo
df = pd.read_excel("jugadores_codificados.xlsx")

# Preparar datos
X = df.drop(columns=["№", "Jugadores", "Posición_delantero", "Posición_mediocampista", "Posición_defensa", "Posición_portero"], errors='ignore')
y = df[["Posición_delantero", "Posición_mediocampista", "Posición_defensa", "Posición_portero"]].idxmax(axis=1)

# Codificación de la variable objetivo
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Separar datos
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
acc_rf = accuracy_score(y_test, rf.predict(X_test))

# Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
acc_lr = accuracy_score(y_test, lr.predict(X_test))

# XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', objective='multi:softprob', num_class=4)
xgb_model.fit(X_train, y_train)
acc_xgb = accuracy_score(y_test, xgb_model.predict(X_test))

# Mostrar resultados
print("Logistic Regression Accuracy:", round(acc_lr, 4))
print("Random Forest Accuracy:", round(acc_rf, 4))
print("XGBoost Accuracy:", round(acc_xgb, 4))
