#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 16:52:09 2026

@author: Estudiante
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Pandas detecta el .gz automáticamente
df = pd.read_csv('letras.csv.gz') 

# Cantidad de datos y clases (Punto 1)
print(f"Instancias: {df.shape[0]}, Atributos: {df.shape[1]}") # 
print(f"Letras únicas: {df['label'].nunique()}") #

# Separar la variable de interés del resto
y = df.iloc[:, 0]      # Suponiendo que la letra es la primera columna
X = df.iloc[:, 1:]     # Los 784 píxeles restantes



# 1. Separar los píxeles (X) de la letra (y)
X = df.iloc[:, 1:] 

# 2. Calcular la varianza de cada píxel (columna)
varianzas = X.var()

# 3. Rearmar la varianza como una imagen de 28x28
mapa_varianza = np.array(varianzas).reshape((28, 28))

# 4. Graficar
plt.figure(figsize=(8, 6))
plt.imshow(mapa_varianza, cmap='hot')
plt.colorbar(label='Varianza')
plt.title("Relevancia de Atributos (Píxeles) según su Varianza")
plt.show()






# 1. Calculamos las varianzas de los 784 píxeles
varianzas = X.var()

# 2. Definimos un umbral basado en un percentil (ej: el 10% más bajo)
umbral = varianzas.quantile(0.10) 

# 3. Identificamos las columnas que superan ese umbral
columnas_relevantes = varianzas[varianzas > umbral].index

# 4. Filtramos el DataFrame
X_reducido = X[columnas_relevantes]
print(f"Atributos originales: {X.shape[1]}")

print(f"Atributos descartados: {len(X.columns) - len(X_reducido.columns)}")



# Contar cuántas veces aparece cada letra (asumiendo que la columna 0 es 'label')
counts = df.iloc[:, 0].value_counts().sort_index()

# Graficar
plt.figure(figsize=(10, 5))
sns.barplot(x=counts.index, y=counts.values, palette="viridis")
plt.title("Distribución de Clases (Letras A-Z)")
plt.xlabel("Letra")
plt.ylabel("Cantidad de Imágenes")
plt.show()

# Imprimir el mínimo y máximo para el informe
print(f"Clase con más datos: {counts.max()}")
print(f"Clase con menos datos: {counts.min()}")

# 1. Verificar nulos
nulos_totales = df.isnull().sum().sum()
print(f"Cantidad de valores nulos: {nulos_totales}")

# 2. Rango de los píxeles (debería ser entre 0 y 255)
valor_min = df.iloc[:, 1:].values.min()
valor_max = df.iloc[:, 1:].values.max()
print(f"Rango de valores de píxeles: [{valor_min}, {valor_max}]")
print('es de 0 a 255')

# Chequeo de filas duplicadas (Otra característica relevante)
duplicados = df.duplicated().sum()
print(f"Cantidad de filas duplicadas: {duplicados}")