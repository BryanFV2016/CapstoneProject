import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo Excel
df = pd.read_excel("jugadores_codificados.xlsx")

# Contar cuántos jugadores hay por cada posición
conteo_posiciones = {
    "Delantero": df["Posición_delantero"].sum(),
    "Mediocampista": df["Posición_mediocampista"].sum(),
    "Defensa": df["Posición_defensa"].sum(),
    "Portero": df["Posición_portero"].sum()
}

# Crear gráfico de barras
plt.figure(figsize=(8, 4))
bars = plt.bar(conteo_posiciones.keys(), conteo_posiciones.values(), color="mediumseagreen", edgecolor="black")

# Añadir etiquetas numéricas encima de cada barra
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height * 0.5,  # Posición más baja (dentro de la barra)
        str(int(height)),
        ha='center', va='center',
        fontsize=10,
        color='white',  # Texto blanco dentro de barra verde
        fontweight='bold'
    )

# Personalización del gráfico
plt.title("Distribution of players by position")
plt.xlabel("Position")
plt.ylabel("Number of players")
plt.ylim(0, max(conteo_posiciones.values()) + 5)
plt.tight_layout()
plt.show()
