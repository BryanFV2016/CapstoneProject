import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np

# --- 1. CORRELATION HEATMAP ---

# Load data
df = pd.read_excel("jugadores_codificados.xlsx")

# Drop non-numeric columns
df_corr = df.drop(columns=['№', 'Jugadores'], errors='ignore')

# Compute correlation matrix
corr_matrix = df_corr.corr(numeric_only=True)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Map Between Morphological Variables")
plt.tight_layout()
plt.show()

# --- 2. MLP TRAINING AND LOSS / ACCURACY PLOTS ---

# Prepare data
X = df_corr.drop(columns=['Posición_delantero', 'Posición_mediocampista', 'Posición_defensa', 'Posición_portero'], errors='ignore')
y = df_corr[['Posición_delantero', 'Posición_mediocampista', 'Posición_defensa', 'Posición_portero']].idxmax(axis=1)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Define MLP model
model = Sequential()
model.add(Dense(32, input_dim=X.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=0)

# Plot training vs validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve of the MLP Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Plot training vs validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve of the MLP Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# --- 3. XGBOOST TRAINING AND METRICS ---

# Prepare numeric labels
y_numeric = label_encoder.fit_transform(y)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_numeric, test_size=0.2, random_state=42)

# Train XGBoost with evaluation metrics
eval_set = [(X_train2, y_train2), (X_test2, y_test2)]
model_xgb = xgb.XGBClassifier(num_class=4, objective='multi:softprob', eval_metric=["mlogloss", "merror"], use_label_encoder=False)
history_xgb = model_xgb.fit(X_train2, y_train2, eval_set=eval_set, verbose=False)

# Get evaluation results
results = model_xgb.evals_result()

# Plot log loss
plt.figure(figsize=(8, 5))
plt.plot(results['validation_0']['mlogloss'], label='Training Log Loss')
plt.plot(results['validation_1']['mlogloss'], label='Validation Log Loss')
plt.title('Log Loss of the XGBoost Model')
plt.xlabel('Iterations')
plt.ylabel('Log Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Plot error rate
plt.figure(figsize=(8, 5))
plt.plot(results['validation_0']['merror'], label='Training Error')
plt.plot(results['validation_1']['merror'], label='Validation Error')
plt.title('Classification Error of the XGBoost Model')
plt.xlabel('Iterations')
plt.ylabel('Error Rate')
plt.legend()
plt.tight_layout()
plt.show()
