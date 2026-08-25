import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

RUTA_DATOS = "data/raw/spam.csv"
RUTA_MODELO = "modelos/spam_v1.pkl"


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


# ---------------------------------------------------------------
# 4. Construir el Pipeline
#
# TfidfVectorizer convierte texto -> numeros
#   lowercase=True    -> "FREE" y "free" son la misma palabra
#   ngram_range=(1,2) -> aprende palabras sueltas Y pares de
#                        palabras. "free entry" como par es mas
#                        informativo que "free" y "entry" sueltas.
#   min_df=2          -> ignora palabras que aparecen en un solo
#                        mensaje. Reduce ruido y tamano del modelo.
#   sublinear_tf=True -> amortigua la frecuencia. Que una palabra
#                        aparezca 10 veces no la hace 10x mas
#                        importante que si apareciera 1 vez.
#
# LogisticRegression aprende un peso por palabra
#   max_iter=1000       -> iteraciones para que el ajuste converja
#   class_weight        -> compensa el desbalance 87/13 dandole mas
#     ="balanced"          importancia a los errores sobre la clase
#                          minoritaria (spam)
# ---------------------------------------------------------------
modelo = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )),
])


# ---------------------------------------------------------------
# 5. Entrenar
# ---------------------------------------------------------------
separador("3. ENTRENAMIENTO")
modelo.fit(X_train, y_train)
vocabulario = modelo.named_steps["tfidf"].vocabulary_
print(f"Listo. Vocabulario aprendido: {len(vocabulario)} terminos")


# ---------------------------------------------------------------
# 6. Evaluar sobre datos que el modelo NUNCA vio
# ---------------------------------------------------------------
y_pred = modelo.predict(X_test)

separador("4. RESULTADOS")
print(f"Exactitud (accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nReporte por clase:")
print(classification_report(y_test, y_pred, digits=3))

separador("5. MATRIZ DE CONFUSION")
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
print("                  PREDICHO ham   PREDICHO spam")
print(f"  REAL ham             {cm[0][0]:5d}          {cm[0][1]:5d}")
print(f"  REAL spam            {cm[1][0]:5d}          {cm[1][1]:5d}")
print(f"\n  Falsos positivos (mensaje bueno marcado spam): {cm[0][1]}")
print(f"  Falsos negativos (spam que se cuela):          {cm[1][0]}")


# ---------------------------------------------------------------
# 7. Que palabras aprendio? (interpretabilidad)
# ---------------------------------------------------------------
separador("6. PALABRAS MAS INDICATIVAS DE SPAM")
terminos = modelo.named_steps["tfidf"].get_feature_names_out()
pesos = modelo.named_steps["clf"].coef_[0]
ranking = sorted(zip(terminos, pesos), key=lambda p: p[1], reverse=True)

print("Mayor peso hacia SPAM:")
for termino, peso in ranking[:12]:
    print(f"  {peso:+.3f}  {termino}")

print("\nMayor peso hacia NORMAL:")
for termino, peso in ranking[-8:]:
    print(f"  {peso:+.3f}  {termino}")


# ---------------------------------------------------------------
# 8. Guardar el pipeline completo
# ---------------------------------------------------------------
os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
joblib.dump(modelo, RUTA_MODELO)

separador("7. MODELO GUARDADO")
print(f"Archivo: {RUTA_MODELO}")
print("Contiene el vectorizador Y el clasificador juntos.")


# ---------------------------------------------------------------
# 9. Prueba manual
# ---------------------------------------------------------------
separador("8. PRUEBA CON MENSAJES NUEVOS")
ejemplos = [
    "Hey, are we still meeting at 5?",
    "CONGRATULATIONS! You have WON a FREE prize. Call 09061701461 now to claim",
    "Can you pick up milk on your way home",
    "URGENT! Your account will be suspended. Click here to verify",
]

for texto in ejemplos:
    pred = modelo.predict([texto])[0]
    prob = modelo.predict_proba([texto])[0].max()
    print(f"  [{pred:>4}] {prob:.1%}  {texto[:52]}")