import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Leer el archivo con los F1 por clase
df_f1 = pd.read_excel("jugadores_codificados.xlsx", sheet_name="f1_por_clase")

# Asegurar que esté ordenado como se desea
df_f1 = df_f1.sort_values(by="F1-Score", ascending=False)

# Crear gráfico
plt.figure(figsize=(8, 5))
sns.barplot(data=df_f1, x="Clase", y="F1-Score", palette="crest")
plt.ylim(0, 1)
plt.title("F1-Score by Class - XGBoost Final Model")
plt.ylabel("F1-Score")
plt.xlabel("Clase")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
