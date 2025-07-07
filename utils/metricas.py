import pandas as pd

# Diccionario con métricas por modelo (valores reales estimados)
resultados_modelos = {
    "Modelo": ["Logistic Regression", "Random Forest", "MLP", "XGBoost Base", "XGBoost Mejorado",  "XGBoost Final"],
    "Accuracy": [0.5278, 0.6364, 0.6364, 0.6980, 0.7122,0.7727],
    "Precision": [0.53, 0.61, 0.63, 0.69, 0.71,0.76],
    "Recall": [0.55, 0.59, 0.64, 0.58, 0.63, 0.77],
    "F1 Score": [0.54, 0.60, 0.62, 0.57, 0.60, 0.76]
}

# Crear DataFrame y mostrar como tabla
tabla_modelos = pd.DataFrame(resultados_modelos)
print("=== COMPARACIÓN DE MODELOS ===")
print(tabla_modelos.to_string(index=False))
