import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import xgboost as xgb

df = pd.read_excel("jugadores_codificados.xlsx")

X = df.drop(columns=["№", "Jugadores", "Posición_delantero", "Posición_mediocampista", "Posición_defensa", "Posición_portero"])
y = df[["Posición_delantero", "Posición_mediocampista", "Posición_defensa", "Posición_portero"]].idxmax(axis=1)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
acc_rf_train = accuracy_score(y_train, rf.predict(X_train))
acc_rf_test = accuracy_score(y_test, rf.predict(X_test))

lr = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='multinomial')
lr.fit(X_train, y_train)
acc_lr_train = accuracy_score(y_train, lr.predict(X_train))
acc_lr_test = accuracy_score(y_test, lr.predict(X_test))

# === XGBOOST ===
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', objective='multi:softprob', num_class=4)
xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
xgb_eval = xgb_model.evals_result()

# === MLP ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_cat = to_categorical(y_encoded)
X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42)

mlp = Sequential([
    Dense(32, activation='relu', input_shape=(X.shape[1],)),
    Dense(16, activation='relu'),
    Dense(4, activation='softmax')
])
mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_mlp = mlp.fit(X_train_mlp, y_train_mlp, epochs=50, validation_data=(X_test_mlp, y_test_mlp), verbose=0)

# === FUNCIÓN PARA GRAFICAR ===
def graficar_metricas(titulo, train_acc, val_acc, train_loss, val_loss, filename):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(titulo)

    axs[0].plot(train_acc, label="Training Accuracy", color="blue")
    axs[0].plot(val_acc, label="Validation Accuracy", color="orange")
    axs[0].set_title("Accuracy")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(train_loss, label="Training Loss", color="green")
    axs[1].plot(val_loss, label="Validation Loss", color="red")
    axs[1].set_title("Loss")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close()

# === GUARDAR GRAFICOS ===
graficar_metricas(
    "XGBoost - Training and Validation",
    xgb_eval["validation_0"]["mlogloss"],
    xgb_eval["validation_1"]["mlogloss"],
    xgb_eval["validation_0"]["mlogloss"],
    xgb_eval["validation_1"]["mlogloss"],
    "xgboost_metrics.png"
)

graficar_metricas(
    "MLP - Training and Validation",
    history_mlp.history["accuracy"],
    history_mlp.history["val_accuracy"],
    history_mlp.history["loss"],
    history_mlp.history["val_loss"],
    "mlp_metrics.png"
)

# === BARPLOT DE MODELOS CLÁSICOS ===
plt.figure(figsize=(8, 4))
plt.bar(["RF Train", "RF Test", "LR Train", "LR Test"],
        [acc_rf_train, acc_rf_test, acc_lr_train, acc_lr_test],
        color=["steelblue", "orange", "seagreen", "tomato"])
plt.title("Accuracy of Random Forest and Logistic Regression")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("classic_models_accuracy.png")
plt.close()
