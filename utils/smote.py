import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# --- CARGAR Y PREPARAR DATOS ---
df = pd.read_excel("jugadores_codificados.xlsx")
pos_cols = [col for col in df.columns if col.startswith('Posición_')]
X = df.drop(columns=['№', 'Jugadores'] + pos_cols)
y = df[pos_cols].idxmax(axis=1)

# --- NORMALIZAR ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- APLICAR SMOTE ---
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# --- DIVIDIR EN ENTRENAMIENTO Y PRUEBA ---
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
