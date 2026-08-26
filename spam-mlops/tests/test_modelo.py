"""
Pruebas del detector de spam, en tres categorias que responden
preguntas distintas:

  A) COMPORTAMIENTO -> el modelo acierta en casos obvios?
  B) CONTRATO        -> la API responde con el formato prometido?
  C) CALIDAD         -> el modelo supera los umbrales de negocio?

A y B corren siempre (local o CI): usan contexto_modelo, que cae de
vuelta a entrenar un pipeline sobre tests/fixtures/muestra_spam.csv
si el registro de MLflow no esta disponible. Solo las dos pruebas de
integracion al final, que validan el registro REAL sin ese fallback,
llevan @pytest.mark.requiere_modelo.
"""

import math
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src import api

RUTA_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "muestra_spam.csv"

# -----------------------------------------------------------------
# Umbrales de calidad. Son decisiones de NEGOCIO -que tan confiable
# tiene que ser el modelo para servir mensajes reales de usuarios-,
# no numeros que se deduzcan del algoritmo. Por eso viven aqui como
# constantes con nombre, para que cambiarlos sea una decision
# explicita y visible, no un numero perdido dentro de un assert.
# -----------------------------------------------------------------
EXACTITUD_MINIMA = 0.95
PRECISION_SPAM_MINIMA = 0.90
RECALL_SPAM_MINIMO = 0.80
VENTAJA_MINIMA_SOBRE_TRIVIAL = 0.05  # puntos de accuracy sobre "siempre ham"

MENSAJES_SPAM_OBVIO = [
    "WINNER!! Claim your FREE prize now call 09061701461",
    "URGENT! Your account will be suspended. Click here to verify http://bit.ly/xyz",
    "Congratulations, you have won a $1000 gift card. Text CLAIM to 82323 now",
]

MENSAJES_HAM_OBVIO = [
    "Hey, are we still meeting at 5?",
    "Can you pick up milk on your way home",
    "im running late, traffic is crazy. be there in 15",
]


# ===================================================================
# Fixtures compartidas
# ===================================================================
@pytest.fixture(scope="session")
def contexto_modelo():
    """
    Da un modelo entrenado mas un conjunto de evaluacion, sin
    importar si hay un registro de MLflow disponible.

    - Si el modelo de produccion esta registrado (entorno local con
      mlflow.db): se usa ese modelo real, evaluado contra el fixture
      completo. Nota: el fixture viene del mismo dataset original,
      asi que una parte pudo haber sido vista en entrenamiento; esto
      funciona como humo de calidad sobre el modelo real, no como
      benchmark riguroso con datos 100% nunca vistos.
    - Si no (CI, sin mlflow.db ni mlruns/): se entrena un pipeline
      TF-IDF + Naive Bayes -misma familia que la configuracion
      "naive-bayes" de entrenar.py- sobre un split propio del
      fixture, y se evalua sobre el 20% que ese entrenamiento nunca
      vio. min_df=1 (en vez del min_df=2 de produccion) porque con
      solo ~960 mensajes de entrenamiento, exigir que un termino
      aparezca dos veces descarta demasiado vocabulario util (ver
      tests/fixtures/generar_muestra.py sobre por que el fixture
      tiene 1200 mensajes y no menos).
    """
    df = pd.read_csv(RUTA_FIXTURE)

    try:
        modelo = mlflow.sklearn.load_model(f"models:/{api.NOMBRE_MODELO}@{api.ALIAS_MODELO}")
        origen = "registro"
        X_eval, y_eval = df["mensaje"], df["tipo"]
    except Exception:
        # Distintas formas de fallar segun el entorno (sin
        # mlflow.db, con mlflow.db pero sin el alias, etc.): todas
        # significan lo mismo aqui, "no hay registro usable", y
        # todas deben caer al mismo fallback.
        origen = "fixture"
        X_train, X_test, y_train, y_test = train_test_split(
            df["mensaje"], df["tipo"], test_size=0.2, random_state=42, stratify=df["tipo"]
        )
        modelo = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", MultinomialNB(alpha=0.05)),
        ])
        modelo.fit(X_train, y_train)
        X_eval, y_eval = X_test, y_test

    return SimpleNamespace(modelo=modelo, X_eval=X_eval, y_eval=y_eval, origen=origen)


@pytest.fixture(scope="session")
def metricas(contexto_modelo):
    y_pred = contexto_modelo.modelo.predict(contexto_modelo.X_eval)
    return {
        "accuracy": accuracy_score(contexto_modelo.y_eval, y_pred),
        "precision_spam": precision_score(
            contexto_modelo.y_eval, y_pred, pos_label="spam", zero_division=0
        ),
        "recall_spam": recall_score(
            contexto_modelo.y_eval, y_pred, pos_label="spam", zero_division=0
        ),
    }


@pytest.fixture
def cliente_api(monkeypatch, contexto_modelo):
    """
    TestClient de la API que no depende de que el registro real de
    MLflow este disponible: se reemplaza cargar_modelo() (la funcion
    que el lifespan invoca al arrancar) por una version que devuelve
    el modelo ya resuelto en contexto_modelo. Asi el contrato de la
    API se puede validar igual en CI que en local.
    """

    def _cargar_modelo_de_prueba():
        return contexto_modelo.modelo, "prueba", datetime.now(timezone.utc)

    monkeypatch.setattr(api, "cargar_modelo", _cargar_modelo_de_prueba)

    with TestClient(api.app) as client:
        yield client


# ===================================================================
# A) COMPORTAMIENTO: el modelo acierta en casos obvios
# ===================================================================
@pytest.mark.parametrize("mensaje", MENSAJES_SPAM_OBVIO)
def test_reconoce_spam_evidente(contexto_modelo, mensaje):
    assert contexto_modelo.modelo.predict([mensaje])[0] == "spam"


@pytest.mark.parametrize("mensaje", MENSAJES_HAM_OBVIO)
def test_reconoce_ham_evidente(contexto_modelo, mensaje):
    assert contexto_modelo.modelo.predict([mensaje])[0] == "ham"


def test_invariante_a_mayusculas(contexto_modelo):
    # TfidfVectorizer normaliza a minusculas antes de vectorizar
    # (lowercase=True es su valor por defecto, y ni entrenar.py ni
    # el fallback lo desactivan), asi que el resultado no deberia
    # cambiar solo porque el mensaje llegue en otro casing.
    mensaje = "WINNER!! Claim your FREE prize now call 09061701461"
    prediccion_original = contexto_modelo.modelo.predict([mensaje])[0]
    prediccion_minusculas = contexto_modelo.modelo.predict([mensaje.lower()])[0]
    assert prediccion_original == prediccion_minusculas


def test_probabilidades_son_una_distribucion_valida(contexto_modelo):
    probabilidades = contexto_modelo.modelo.predict_proba(["mensaje cualquiera de prueba"])[0]
    assert math.isclose(probabilidades.sum(), 1.0, rel_tol=1e-6)
    assert all(0.0 <= p <= 1.0 for p in probabilidades)


# ===================================================================
# B) CONTRATO: la API cumple su formato de entrada/salida
# ===================================================================
def test_salud_responde_ok(cliente_api):
    respuesta = cliente_api.get("/salud")
    cuerpo = respuesta.json()
    assert respuesta.status_code == 200
    assert cuerpo["estado"] == "ok"
    assert cuerpo["version"] is not None


def test_predecir_devuelve_los_campos_del_contrato(cliente_api):
    respuesta = cliente_api.post("/predecir", json={"mensaje": "Hey, are we still meeting at 5?"})
    cuerpo = respuesta.json()
    assert respuesta.status_code == 200
    assert set(cuerpo.keys()) == {"etiqueta", "probabilidad_spam", "version_modelo"}
    assert cuerpo["etiqueta"] in ("spam", "ham")


def test_mensaje_vacio_devuelve_422(cliente_api):
    respuesta = cliente_api.post("/predecir", json={"mensaje": ""})
    assert respuesta.status_code == 422


def test_campo_incorrecto_devuelve_422(cliente_api):
    # El contrato exige el campo "mensaje"; "texto" no existe para
    # el modelo Pydantic y FastAPI debe rechazarlo explicitamente,
    # no ignorarlo silenciosamente.
    respuesta = cliente_api.post("/predecir", json={"texto": "hola"})
    assert respuesta.status_code == 422


def test_lote_devuelve_una_prediccion_por_mensaje(cliente_api):
    mensajes = [
        "Hey, are we still meeting at 5?",
        "WINNER!! Claim your FREE prize now call 09061701461",
    ]
    respuesta = cliente_api.post("/predecir-lote", json={"mensajes": mensajes})
    cuerpo = respuesta.json()
    assert respuesta.status_code == 200
    assert len(cuerpo["predicciones"]) == len(mensajes)


# ===================================================================
# C) CALIDAD: el modelo supera los umbrales estadisticos de negocio
# ===================================================================
def test_exactitud_minima(metricas):
    assert metricas["accuracy"] >= EXACTITUD_MINIMA


def test_precision_spam_minima(metricas):
    assert metricas["precision_spam"] >= PRECISION_SPAM_MINIMA


def test_recall_spam_minimo(metricas):
    assert metricas["recall_spam"] >= RECALL_SPAM_MINIMO


def test_supera_al_clasificador_trivial(contexto_modelo, metricas):
    # "Siempre ham" acierta ~87% solo por el desbalance de clases.
    # Un modelo que no le gane con margen no esta aportando nada,
    # solo esta memorizando la clase mayoritaria.
    accuracy_trivial = (contexto_modelo.y_eval == "ham").mean()
    assert metricas["accuracy"] - accuracy_trivial >= VENTAJA_MINIMA_SOBRE_TRIVIAL


# ===================================================================
# Integracion real: a diferencia de todo lo anterior, esto SI
# depende de que el alias "produccion" resuelva en el mlflow.db
# local -no hay fallback posible, porque si no hay registro no hay
# nada real que validar-. Por eso llevan requiere_modelo y quedan
# fuera del comando de CI (pytest -m "not requiere_modelo").
# ===================================================================
@pytest.mark.requiere_modelo
def test_alias_produccion_resuelve_en_el_registro_real():
    modelo = mlflow.sklearn.load_model(f"models:/{api.NOMBRE_MODELO}@{api.ALIAS_MODELO}")
    assert hasattr(modelo, "predict")


@pytest.mark.requiere_modelo
def test_api_real_carga_el_modelo_de_produccion():
    with TestClient(api.app) as cliente:
        respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json()["alias"] == api.ALIAS_MODELO
