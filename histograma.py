import pandas as pd
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_excel("jugadores_codificados.xlsx")

# Histograma de IMC (puedes cambiar 'IMC' por 'Altura' o 'Peso')
plt.figure(figsize=(8, 5))
plt.hist(df["IMC"], bins=15, color='skyblue', edgecolor='black')
plt.title("Distribution of BMI (Body Mass Index) among Players", fontsize=14)
plt.xlabel("BMI", fontsize=12)
plt.ylabel("Number of Players", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("histogram_bmi.png")  # Opcional: guarda el gráfico como imagen
plt.show()
