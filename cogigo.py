#Tarea de Recuperación-Evaluación 2
#Fundamentos de Data Science 

import os 
import numpy as np 
import pandas as pd 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay)

# Configuracion general y reproducibilidad.
SEED = 42
np.random.seed(SEED)

RUTA_DATOS = "base_riesgo_crediticio.csv" CARPETA_RESULTADOS = "resultados" os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

# Carga y exploracion de datos (EDA) basica
df = pd.read_csv(RUTA_DATOS)

print("=" * 60)
print("1.EXPLORACION DE DATOS (EDA)")
print("=" * 60)
print(f"Dimensiones del dataset: {df.shape}")
print("\nTipos de datos:")
print(df.dtypes)
print("\Valores nulos por columna:")
print(df.isnull().sum())
print("\nEstadisticas descriptivas:)
