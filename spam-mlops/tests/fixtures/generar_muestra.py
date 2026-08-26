"""
Genera tests/fixtures/muestra_spam.csv: una muestra reducida del
dataset real que SI se commitea a Git (el dataset completo esta
versionado con DVC, no con Git). Sirve para que las pruebas de
calidad puedan entrenar y evaluar un modelo al vuelo en CI, donde
mlflow.db no esta disponible.

Uso (desde spam-mlops/): python tests/fixtures/generar_muestra.py
"""

import pandas as pd

SEMILLA = 42
# 400 mensajes (el tamano inicial que se probo) no le daba a un
# Naive Bayes entrenado al vuelo señal suficiente para generalizar a
# frases de spam poco frecuentes en el dataset (ej. "account
# suspended": la palabra "suspend" no aparece ni una sola vez en las
# 5414 filas del dataset completo). 1200 es el tamano mas chico con
# el que las pruebas de comportamiento (tests/test_modelo.py, seccion
# A) pasan de forma consistente contra el modelo entrenado al vuelo.
N_TOTAL = 1200
RUTA_ORIGEN = "data/raw/spam_limpio.csv"
RUTA_SALIDA = "tests/fixtures/muestra_spam.csv"

df = pd.read_csv(RUTA_ORIGEN)[["tipo", "mensaje"]]

# ---------------------------------------------------------------
# Muestreo estratificado: cada clase se toma en la misma proporcion
# que tiene en el dataset completo (87/13 ham/spam). Si se
# muestreara parejo (200/200), el modelo entrenado sobre este
# fixture aprenderia un desbalance que no existe en produccion y las
# metricas de calidad no significarian lo mismo.
# ---------------------------------------------------------------
proporciones = df["tipo"].value_counts(normalize=True)
partes = []
for clase, proporcion in proporciones.items():
    n_clase = round(N_TOTAL * proporcion)
    partes.append(df[df["tipo"] == clase].sample(n=n_clase, random_state=SEMILLA))

muestra = pd.concat(partes).sample(frac=1, random_state=SEMILLA).reset_index(drop=True)
muestra.to_csv(RUTA_SALIDA, index=False)

print(f"Generados {len(muestra)} mensajes -> {RUTA_SALIDA}")
print(muestra["tipo"].value_counts().to_string())
