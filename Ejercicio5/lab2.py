import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Leer el archivo CSV.
# Asegúrate de que 'weather.csv' esté en el mismo directorio o especifica la ruta correcta.
df = pd.read_csv('weather.csv')

# Las columnas que contienen las temperaturas son "0", "1", ..., "11"
meses = [str(i) for i in range(12)]
X = df[meses].values  # Matriz de tamaño (35, 12)
estaciones = df['station'].values

# 2. Aplicar PCA
# Se seleccionan 2 componentes principales para poder visualizarlos en 2D.
pca = PCA(n_components=2)
scores = pca.fit_transform(X)      # Proyecciones de cada estación en el espacio PC1-PC2 (dimensión: 35 x 2)
componentes = pca.components_        # Matriz de componentes (2 x 12): cada fila es un componente y contiene los "loadings" para cada mes

# Imprimir los componentes para ver sus valores
print("Primer componente (p1):")
print(componentes[0])
print("\nSegundo componente (p2):")
print(componentes[1])

# 3. Graficar las curvas de los dos primeros componentes
plt.figure(figsize=(12,5))

# Graficar p1 (primer componente)
plt.subplot(1,2,1)
plt.plot(range(12), componentes[0], marker='o', color='blue')
plt.xlabel('Mes (índice)')
plt.ylabel('Carga')
plt.title('Primer componente principal (p1)')
plt.xticks(range(12))  # Etiqueta cada mes del 0 al 11

# Graficar p2 (segundo componente)
plt.subplot(1,2,2)
plt.plot(range(12), componentes[1], marker='o', color='green')
plt.xlabel('Mes (índice)')
plt.ylabel('Carga')
plt.title('Segundo componente principal (p2)')
plt.xticks(range(12))

plt.tight_layout()
plt.show()

# 4. Realizar el biplot (representar las estaciones en el espacio PC1 - PC2)
plt.figure(figsize=(10,8))
plt.scatter(scores[:, 0], scores[:, 1], color='red')

# Agregar etiqueta a cada punto (estación)
for i, nombre in enumerate(estaciones):
    plt.text(scores[i, 0] + 0.1, scores[i, 1] + 0.1, nombre, fontsize=9)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Biplot de las estaciones canadienses')
plt.grid(True)
plt.show()
