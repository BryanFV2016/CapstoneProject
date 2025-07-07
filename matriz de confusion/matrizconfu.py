import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import matplotlib.pyplot as plt

# 1. Cargar datos
df = pd.read_excel("jugadores_codificados.xlsx")

# 2. Separar características y etiquetas
X = df.drop(columns=['№', 'Jugadores', 'Posición_delantero', 'Posición_mediocampista',
                     'Posición_defensa', 'Posición_portero'], errors='ignore')

y = df[['Posición_delantero', 'Posición_mediocampista',
        'Posición_defensa', 'Posición_portero']].idxmax(axis=1)

# 3. Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. Entrenar modelo XGBoost final
model_xgb = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
model_xgb.fit(X_train, y_train)

# 6. Predecir
y_pred = model_xgb.predict(X_test)

# 7. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

# 8. Mostrar gráfico
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Final XGBoost Model")
plt.grid(False)
plt.tight_layout()
plt.show()
