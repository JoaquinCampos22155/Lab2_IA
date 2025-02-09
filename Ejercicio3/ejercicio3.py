import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_1samp

# Obtener la ruta absoluta del archivo
script_dir = os.path.dirname(os.path.abspath(__file__))  
file_path = os.path.join(script_dir, "areas.csv")

# Verificar si el archivo existe antes de leerlo
if not os.path.exists(file_path):
    print(f"Error: No se encontró el archivo en {file_path}")
    exit()

df = pd.read_csv(file_path, delimiter=';')

# Extraer el primer dígito de cada valor en la columna "Área en kilómetros cuadrados"
def first_digit(n):
    while n >= 10:
        n //= 10
    return n

df['First Digit'] = df['Area in square kilometres'].astype(int).apply(first_digit)

# Contar la frecuencia de cada primer dígito
digit_counts = df['First Digit'].value_counts(normalize=True).sort_index()

# Distribución teórica según la Ley de Benford
benford_dist = {d: np.log10(1 + 1/d) for d in range(1, 10)}

# Crear la función de distribución acumulativa de Benford
benford_cdf_values = np.array([np.sum(list(benford_dist.values())[:d]) for d in range(1, 10)])

# Transformar los datos observados en una distribución acumulativa
empirical_cdf_values = np.array([(df['First Digit'] <= d).mean() for d in range(1, 10)])

# Aplicar la prueba de Kolmogorov-Smirnov
ks_stat, ks_p_value = ks_1samp(df['First Digit'], lambda x: np.searchsorted(list(benford_dist.keys()), x, side='right') / 9)

# Visualizar los resultados
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(digit_counts.index, digit_counts.values, alpha=0.7, label="Datos observados")
ax.plot(list(benford_dist.keys()), list(benford_dist.values()), marker="o", linestyle="dashed", color="red", label="Ley de Benford")
ax.set_xticks(range(1, 10))
ax.set_xlabel("Primer dígito")
ax.set_ylabel("Frecuencia relativa")
ax.set_title("Distribución del primer dígito vs. Ley de Benford")
ax.legend()
plt.show()

# Mostrar resultados de la prueba KS
print(f"Estadístico KS: {ks_stat:.5f}")
print(f"Valor p: {ks_p_value:.5e}")

# Interpretación del resultado
if ks_p_value < 0.05:
    print("Los datos NO siguen la Ley de Benford (se rechaza H0).")
else:
    print("Los datos siguen la Ley de Benford (no se rechaza H0).")
