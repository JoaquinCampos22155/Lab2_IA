import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Definir los parámetros de la distribución
n = 4  # Dimensión
mu = np.array([5, 10, 15, 20])  # Vector de medias
cov_matrix = np.array([
    [1.0, 0.5, 0.3, 0.2], 
    [0.5, 1.5, 0.6, 0.4], 
    [0.3, 0.6, 2.0, 0.5], 
    [0.2, 0.4, 0.5, 1.2]
])

# Generar la muestra aleatoria
num_samples = 1000  # Número de muestras
samples = np.random.multivariate_normal(mu, cov_matrix, size=num_samples)

# Convertir los datos en un DataFrame para visualización
df = pd.DataFrame(samples, columns=[f'Var{i+1}' for i in range(n)])

# Graficar el pairplot
sns.pairplot(df, diag_kind="kde")
plt.suptitle("Distribución de la Muestra Gaussiana Multivariada", y=1.02)
plt.show()

# Calcular las estadísticas muestrales
sample_mean = np.mean(samples, axis=0)
sample_cov = np.cov(samples, rowvar=False)

# Mostrar los resultados
print("Vector de medias teóricas:")
print(mu)
print("\nVector de medias muestrales:")
print(sample_mean)

print("\nMatriz de covarianza teórica:")
print(cov_matrix)
print("\nMatriz de covarianza muestral:")
print(sample_cov)

# Verificar si los valores muestrales son similares a los teóricos
diff_mean = np.abs(mu - sample_mean)
diff_cov = np.abs(cov_matrix - sample_cov)

print("\nDiferencia entre medias teóricas y muestrales:")
print(diff_mean)

print("\nDiferencia entre covarianzas teóricas y muestrales:")
print(diff_cov)
