#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 16:28:34 2026

@author: Estudiante
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Carga de datos
df = pd.read_csv('letras.csv')

# 2. Análisis básico (Punto 1)
print(f"Cantidad de datos: {len(df)}")
print(f"Letras presentes: {df['label'].unique()}") # Asumiendo que la columna se llama 'label'

# 3. Visualizar una letra (Punto 1.c)
def plot_letra(indice):
    # El código de la consigna: extrae píxeles y reshapes a 28x28
    img = np.array(df.iloc[indice, 1:]).reshape((28,28)) 
    plt.imshow(img, cmap='gray')
    plt.title(f"Letra: {df.iloc[indice, 0]}")
    plt.show()

for i in range(1,26000,1016):
    plot_letra(i) # Cambiá el número para ver distintas imágenes