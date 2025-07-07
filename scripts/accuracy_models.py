import matplotlib.pyplot as plt

# Resultados reales de tus modelos (ajusta si los actualizas)
accuracies = {
    'Logistic Regression': 0.53,
    'Random Forest': 0.59,
    'MLP': 0.64,
    'XGBoost': 0.70
}

names = list(accuracies.keys())
values = list(accuracies.values())

# Crear gráfico de barras
plt.figure(figsize=(8, 5))
bars = plt.bar(names, values, color=['orange', 'skyblue', 'mediumseagreen', 'tomato'])
plt.ylim(0.4, 0.8)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Model Accuracy Comparison", fontsize=14)

# Mostrar valor encima de cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}", ha='center', fontsize=11)

plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.show()
