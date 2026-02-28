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


#1.a)
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

#%% 1.b)
# Diccionario de mapeo para no marearse con los números
mapping = {12: 'M', 14: 'O', 16: 'Q', 18: 'S'}

def analizar_similitud(id1, id2):
    # 1. Filtramos por el número de label y calculamos el promedio
    # Usamos .iloc[:, 1:] para no promediar la columna del label
    img1 = df[df['label'] == id1].iloc[:, 1:].mean().values.reshape(28, 28)
    img2 = df[df['label'] == id2].iloc[:, 1:].mean().values.reshape(28, 28)
    
    # 2. Imagen Diferencia (Absoluta)
    diff = np.abs(img1 - img2)
    
    # 3. Graficamos
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img1, cmap='gray'); ax[0].set_title(f"Promedio {mapping[id1]} ({id1})")
    ax[1].imshow(img2, cmap='gray'); ax[1].set_title(f"Promedio {mapping[id2]} ({id2})")
    ax[2].imshow(diff, cmap='hot'); ax[2].set_title(f"Diferencia {mapping[id1]}-{mapping[id2]}")
    plt.show()
    
    # Retornamos la distancia euclídea para el segundo gráfico
    return np.linalg.norm(img1 - img2)

# Ejecutamos las comparaciones
dist_oq = analizar_similitud(14, 16) # O vs Q
dist_sm = analizar_similitud(18, 12) # S vs M

# Gráfico de barras (Segundo tipo de gráfico pedido)
plt.bar(['O vs Q', 'S vs M'], [dist_oq, dist_sm], color=['salmon', 'skyblue'])
plt.title("Distancia Euclídea entre Letras Promedio")
plt.ylabel("Distancia")
plt.show()

#%%1.c) 
import matplotlib.pyplot as plt

# 1. Filtramos todas las imágenes que sean la letra J (label 9)
df_j = df[df['label'] == 9]

# 2. Elegimos 16 imágenes al azar para ver la variedad de tipografías
muestras_j = df_j.sample(16, random_state=42)

# 3. Creamos una grilla de 4x4
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    # Reshape a 28x28 salteando la columna del label
    img = muestras_j.iloc[i, 1:].values.reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.axis('off')

plt.suptitle("Variabilidad de la letra J (Label 9)")
plt.show()
#%% 2.a)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Filtramos las letras O (14) y L (11)
df_binario = df[df['label'].isin([11, 14])]

# 2. Separamos atributos (X) y etiqueta (y)
X = df_binario.iloc[:, 1:] # Los 784 píxeles
y = df_binario['label']    # La letra (11 o 14)

# Separar el 70% para entrenamiento y 30% para testeo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2.b) y 2.c)