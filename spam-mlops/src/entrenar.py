import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

RUTA_DATOS = "data/raw/spam.csv"
RUTA_MODELO = "modelos/spam_{nombre}.pkl"

# ---------------------------------------------------------------
# Configuraciones a comparar en MLflow.
#
# Cada una define un vectorizador (texto -> numeros) y un
# clasificador. Se entrena y evalua un pipeline por configuracion,
# cada uno en su propio run de MLflow, para poder comparar
# accuracy/precision/recall/f1 entre ellas en la UI.
# ---------------------------------------------------------------
CONFIGURACIONES = [
    {
        "nombre": "baseline-unigramas",
        "vectorizador": TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True),
        "clasificador": LogisticRegression(max_iter=1000, class_weight="balanced"),
    },
    {
        "nombre": "tfidf-bigramas",
        "vectorizador": TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        "clasificador": LogisticRegression(max_iter=1000, class_weight="balanced"),
    },
    {
        "nombre": "sin-class-weight",
        "vectorizador": TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        "clasificador": LogisticRegression(max_iter=1000),
    },
    {
        "nombre": "naive-bayes",
        "vectorizador": TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        "clasificador": MultinomialNB(alpha=0.1),
    },
    {
        "nombre": "conteo-simple",
        "vectorizador": CountVectorizer(ngram_range=(1, 2), min_df=2),
        "clasificador": LogisticRegression(max_iter=1000, class_weight="balanced"),
    },
    {
        "nombre": "svm-lineal",
        "vectorizador": TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        "clasificador": LinearSVC(class_weight="balanced"),
    },
]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("spam-mlops")


def separador(titulo):
    print("\n" + "=" * 66)
    print(titulo)
    print("=" * 66)


# ---------------------------------------------------------------
# 1. Cargar y limpiar
# ---------------------------------------------------------------
df = pd.read_csv(RUTA_DATOS, encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["tipo", "mensaje"]
df = df.drop_duplicates().dropna()

separador("1. DATOS")
print(f"Mensajes utiles: {len(df)}")
print(df["tipo"].value_counts().to_string())


# ---------------------------------------------------------------
# 2. Separar X (entrada) e y (respuesta)
# ---------------------------------------------------------------
X = df["mensaje"]
y = df["tipo"]


# ---------------------------------------------------------------
# 3. Dividir en entrenamiento y prueba
#
# test_size=0.2   -> 20% se aparta para el examen final
# random_state=42 -> fija el azar para que la division sea siempre
#                    la misma. Sin esto, cada corrida daria numeros
#                    distintos y no podrias comparar experimentos.
# stratify=y      -> mantiene la proporcion 87/13 en ambos grupos.
#                    Sin esto, por azar el conjunto de prueba podria
#                    quedar con muy poco spam y la evaluacion no
#                    seria representativa.
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

separador("2. DIVISION DE DATOS")
print(f"Entrenamiento: {len(X_train)} mensajes")
print(f"Prueba:        {len(X_test)} mensajes")


ejemplos = [
    "Hey, are we still meeting at 5?",
    "CONGRATULATIONS! You have WON a FREE prize. Call 09061701461 now to claim",
    "Can you pick up milk on your way home",
    "URGENT! Your account will be suspended. Click here to verify",
]


def entrenar_configuracion(nombre, vectorizador, clasificador):
    with mlflow.start_run(run_name=nombre):
        separador(f"CONFIGURACION: {nombre}")

        mlflow.log_param("configuracion", nombre)
        mlflow.log_param("n_mensajes", len(df))
        mlflow.log_params(df["tipo"].value_counts().to_dict())
        mlflow.log_params({
            "test_size": 0.2,
            "random_state": 42,
            "n_entrenamiento": len(X_train),
            "n_prueba": len(X_test),
        })
        mlflow.log_param("vectorizador", type(vectorizador).__name__)
        mlflow.log_params({f"vec_{k}": v for k, v in vectorizador.get_params().items()})
        mlflow.log_param("clasificador", type(clasificador).__name__)
        mlflow.log_params({f"clf_{k}": v for k, v in clasificador.get_params().items()})

        # -----------------------------------------------------------
        # Entrenar
        # -----------------------------------------------------------
        modelo = Pipeline([
            ("tfidf", vectorizador),
            ("clf", clasificador),
        ])
        modelo.fit(X_train, y_train)
        vocabulario = modelo.named_steps["tfidf"].vocabulary_
        print(f"Listo. Vocabulario aprendido: {len(vocabulario)} terminos")
        mlflow.log_metric("tamano_vocabulario", len(vocabulario))

        # -----------------------------------------------------------
        # Evaluar sobre datos que el modelo NUNCA vio
        # -----------------------------------------------------------
        y_pred = modelo.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Exactitud (accuracy): {accuracy:.4f}")
        print("\nReporte por clase:")
        print(classification_report(y_test, y_pred, digits=3))

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=["ham", "spam"], average=None
        )

        cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
        print("                  PREDICHO ham   PREDICHO spam")
        print(f"  REAL ham             {cm[0][0]:5d}          {cm[0][1]:5d}")
        print(f"  REAL spam            {cm[1][0]:5d}          {cm[1][1]:5d}")
        print(f"\n  Falsos positivos (mensaje bueno marcado spam): {cm[0][1]}")
        print(f"  Falsos negativos (spam que se cuela):          {cm[1][0]}")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metrics({
            "precision_ham": precision[0],
            "recall_ham": recall[0],
            "f1_ham": f1[0],
            "precision_spam": precision[1],
            "recall_spam": recall[1],
            "f1_spam": f1[1],
            "verdaderos_negativos": cm[0][0],
            "falsos_positivos": cm[0][1],
            "falsos_negativos": cm[1][0],
            "verdaderos_positivos": cm[1][1],
        })

        # -----------------------------------------------------------
        # Guardar el pipeline completo
        # -----------------------------------------------------------
        ruta_modelo = RUTA_MODELO.format(nombre=nombre)
        os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)
        joblib.dump(modelo, ruta_modelo)
        mlflow.sklearn.log_model(modelo, name="modelo")

        print(f"\nArchivo: {ruta_modelo}")
        print(f"Tambien registrado en MLflow (run_id: {mlflow.active_run().info.run_id})")

        # -----------------------------------------------------------
        # Prueba manual
        # -----------------------------------------------------------
        print("\nPrueba con mensajes nuevos:")
        for texto in ejemplos:
            pred = modelo.predict([texto])[0]
            print(f"  [{pred:>4}] {texto[:52]}")

        return accuracy


# ---------------------------------------------------------------
# 4. Entrenar y evaluar cada configuracion en su propio run
# ---------------------------------------------------------------
resultados = []
for cfg in CONFIGURACIONES:
    accuracy = entrenar_configuracion(cfg["nombre"], cfg["vectorizador"], cfg["clasificador"])
    resultados.append((cfg["nombre"], accuracy))

separador("RESUMEN DE CONFIGURACIONES")
for nombre, accuracy in sorted(resultados, key=lambda r: r[1], reverse=True):
    print(f"  {accuracy:.4f}  {nombre}")
