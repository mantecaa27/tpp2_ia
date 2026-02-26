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

print(X,y)