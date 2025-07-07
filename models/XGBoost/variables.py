import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
file_path = "jugadores_codificados.xlsx"  # asegúrate de tener el archivo en el mismo directorio
df = pd.read_excel(file_path)

# Set the style
sns.set(style="whitegrid")

# Define the variables to plot
variables = [
    'Altura',
    'Peso',
    'Masa Muscular',
    '% Grasa Corporal',
    'Velocidad Sprint (km/h)',
    'Aceleración (m/s²)',
    'Resistencia Aeróbica (VO2max)',
    'Precisión Pase (%)',
    'Técnica Disparo'
]

# Map of Spanish to English variable names for labels
label_map = {
    'Altura': 'Height (cm)',
    'Peso': 'Weight (kg)',
    'Masa Muscular': 'Muscle Mass (kg)',
    '% Grasa Corporal': 'Body Fat (%)',
    'Velocidad Sprint (km/h)': 'Sprint Speed (km/h)',
    'Aceleración (m/s²)': 'Acceleration (m/s²)',
    'Resistencia Aeróbica (VO2max)': 'Aerobic Endurance (VO₂max)',
    'Precisión Pase (%)': 'Pass Accuracy (%)',
    'Técnica Disparo': 'Shooting Technique'
}

# Create the plots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
axes = axes.flatten()

for ax, var in zip(axes, variables):
    sns.histplot(df[var], kde=True, ax=ax, color='skyblue')
    ax.set_title(f'Distribution of {label_map[var]}', fontsize=12)
    ax.set_xlabel(label_map[var], fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)

plt.tight_layout()
plt.show()
