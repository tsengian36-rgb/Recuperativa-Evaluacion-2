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

RUTA_DATOS = "base_riesgo_crediticio.csv" 
CARPETA_RESULTADOS = "resultados" 
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

#1. Carga y exploracion de datos (EDA) basica
df = pd.read_csv(RUTA_DATOS)

print("=" * 60)
print("1.EXPLORACION DE DATOS (EDA)")
print("=" * 60)
print(f"Dimensiones del dataset: {df.shape}")
print("\nTipos de datos:")
print(df.dtypes)
print("\nValores nulos por columna:")
print(df.isnull().sum())
print("\nEstadisticas descriptivas:")
print(df.describe())
print("\nBalance de la variable objetivo (incumplimiento):")
print(df["incumplimiento"].value_counts(normalize=True))

# Guardo un resumen del EDA
with open(os.path.join(CARPETA_RESULTADOS, "eda_resumen.txt"), "w") as f:
  f.write("RESUMEN EDA - Base de riesgo crediticio\n")
  f.write(f"Dimensiones: {df.shape}\n\n")
  f.write("Nulos por columna:\n")
  f.write(str(df.isnull().sum()) + "\n\n")
  f.write("Estadisticas descriptivas:\n")
  f.write(str(df.describe()) + "\n\n")
  f.write("Balance de clases (incumplimiento):\n")
  f.write(str(df["incumplimiento"].value_counts()) + "\n")

#Grafico de correlaciones
plt.figure(figsize=(8, 6))
columnas_numericas = df.drop(columns=["id_cliente"]).columns
sns.heatmap(df[columnas_numericas].corr(),annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de correlacion de variables numericas")
plt.savefig(os.path.join(CARPETA_RESULTADOS, "01_correlaciones.png"))
plt.close()

#Distribucion de la variable
plt.figure(figsize=(5, 4))
sns.countplot(x="incumplimiento", data=df)
plt.title("Distribucion de la variable objetivo (incumplimiento)")
plt.xlabel("Incumplimiento (0 = no, 1 = si)")
plt.ylabel("Cantidad de clientes")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "02_balance_clases.png"))
plt.close()

# 2. Preprocesamiento: seleccion de features, split y escalamiento
X = df.drop(columns=["id_cliente", "incumplimiento"])
y = df["incumplimiento"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print("\n" + "=" * 60)
print("2. PREPROCESAMIENTO")
print("=" * 60)
print(f"Tamano set de entrenamiento: {X_train.shape}")
print(f"Tamano set de prueba: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Diseno y entrenamiento del MLP
modelo = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    max_iter=500,
    early_stopping=True,
    random_state=SEED,
)

modelo.fit(X_train_scaled, y_train)

print("\n" + "=" * 60)
print("3. ENTRENAMIENTO DEL MLP")
print("=" * 60)
print(f"Numero de iteraciones hasta convergencia: {modelo.n_iter_}")
print(f"Valor final de la funcion de perdida (loss): {modelo.loss_:.4f}")

plt.figure(figsize=(6, 4))
plt.plot(modelo.loss_curve_)
plt.title("Curva de perdida durante el entrenamiento del MLP")
plt.xlabel("Iteracion")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "03_curva_perdida.png"))
plt.close()

# 4. Evaluacion del modelo
y_pred = modelo.predict(X_test_scaled)
y_proba = modelo.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
reporte = classification_report(y_test, y_pred, target_names=["No incumple", "Incumple"])
auc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 60)
print("4. EVALUACION DEL MODELO")
print("=" * 60)
print(f"Accuracy en el set de prueba: {acc:.4f}")
print("\nMatriz de confusion:")
print(cm)
print("\nReporte de clasificacion:")
print(reporte)
print(f"\nAUC-ROC: {auc:.4f}")

# Guardar matriz de confusion como grafico
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No incumple", "Incumple"])
disp.plot(cmap="Blues")
plt.title("Matriz de confusion - MLP")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "04_matriz_confusion.png"))
plt.close()

# Curva ROC
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title(f"Curva ROC - MLP (AUC = {auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "05_curva_roc.png"))
plt.close()

# Guardar metricas en un archivo de texto y uno csv
with open(os.path.join(CARPETA_RESULTADOS, "metricas.txt"), "w") as f:
    f.write("RESULTADOS DEL MODELO MLP - Riesgo Crediticio\n")
    f.write("=" * 50 + "\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"AUC-ROC: {auc:.4f}\n\n")
    f.write("Matriz de confusion:\n")
    f.write(str(cm) + "\n\n")
    f.write("Interpretacion:\n")
    f.write(f"  Verdaderos negativos: {cm[0][0]}\n")
    f.write(f"  Falsos positivos: {cm[0][1]}\n")
    f.write(f"  Falsos negativos: {cm[1][0]}\n")
    f.write(f"  Verdaderos positivos: {cm[1][1]}\n\n")
    f.write("Reporte de clasificacion:\n")
    f.write(reporte + "\n")

pd.DataFrame(
    {
        "metrica": ["accuracy", "auc_roc"],
        "valor": [acc, auc],
    }
).to_csv(os.path.join(CARPETA_RESULTADOS, "metricas.csv"), index=False)

print(f"\nGraficos y metricas guardados en la carpeta '{CARPETA_RESULTADOS}/'.")
print("Ejecucion finalizada correctamente.")
