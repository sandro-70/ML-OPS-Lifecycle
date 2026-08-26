"""
API REST para servir el modelo de deteccion de spam.

Uso: uvicorn src.api:app --reload
(ejecutar desde spam-mlops/, igual que entrenar.py, porque las rutas
de datos y de tracking de MLflow son relativas a ese directorio)
"""

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, status
from mlflow import MlflowClient
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spam-api")

# ---------------------------------------------------------------
# Configuracion
#
# Mismo tracking URI que entrenar.py (sqlite:///mlflow.db en
# spam-mlops/) para que la API lea del mismo registro donde se
# publican los modelos entrenados. Nombre y alias configurables por
# variable de entorno para poder apuntar a otro modelo/etapa (por
# ejemplo "staging") sin tocar el codigo.
# ---------------------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
NOMBRE_MODELO = os.environ.get("MODEL_NAME", "spam-detector")
ALIAS_MODELO = os.environ.get("MODEL_ALIAS", "produccion")

RUTA_LOG_PREDICCIONES = Path("data/produccion/predicciones.jsonl")
LONGITUD_MAXIMA_MENSAJE = 2000

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class EstadoModelo:
    """
    Contenedor mutable con el modelo cargado en memoria.

    Se necesita una clase (en vez de variables sueltas) porque
    /recargar debe poder reemplazar el modelo activo sin reiniciar
    el proceso, y varios requests concurrentes leen este mismo
    estado mientras eso pasa.
    """

    def __init__(self):
        self.modelo = None
        self.version = None
        self.cargado_en = None
        self.error = None


estado = EstadoModelo()

# Protege la lectura/escritura de `estado` entre el hilo que atiende
# /recargar y los hilos que atienden /predecir en paralelo. Sin esto
# un request podria leer una version del modelo mientras otro hilo
# esta a la mitad de reemplazarla.
_candado = threading.Lock()


def cargar_modelo():
    """
    Carga el pipeline completo (vectorizador + clasificador) desde
    el Model Registry de MLflow, por alias en vez de por numero de
    version fijo. Asi, promover un modelo nuevo a "produccion" en
    MLflow (mover el alias) es suficiente para que /recargar lo
    recoja, sin cambiar esta URI.
    """
    uri = f"models:/{NOMBRE_MODELO}@{ALIAS_MODELO}"
    modelo = mlflow.sklearn.load_model(uri)

    cliente = MlflowClient()
    version_info = cliente.get_model_version_by_alias(NOMBRE_MODELO, ALIAS_MODELO)

    # MlflowClient devuelve el numero de version como int; se
    # normaliza a str porque en la API es un identificador, no una
    # cantidad (no tiene sentido sumarlo o compararlo numericamente).
    return modelo, str(version_info.version), datetime.now(timezone.utc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carga unica al arrancar. Si el registro no tiene el modelo o
    # el alias no existe, el servidor sigue levantando (para poder
    # diagnosticar via /salud) pero /predecir respondera 503 hasta
    # que se resuelva con /recargar.
    try:
        modelo, version, cargado_en = cargar_modelo()
        with _candado:
            estado.modelo = modelo
            estado.version = version
            estado.cargado_en = cargado_en
            estado.error = None
        logger.info("Modelo %s@%s version %s cargado", NOMBRE_MODELO, ALIAS_MODELO, version)
    except Exception as exc:
        estado.error = str(exc)
        logger.error("No se pudo cargar el modelo al arrancar: %s", exc)

    yield
    # No hay recursos que liberar: mlflow no mantiene conexiones
    # abiertas de larga duracion para un modelo ya cargado en memoria.


app = FastAPI(
    title="API de deteccion de spam",
    description="Sirve el modelo spam-detector registrado en MLflow.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------
# Modelos Pydantic (entrada/salida)
# ---------------------------------------------------------------
class MensajeEntrada(BaseModel):
    mensaje: str = Field(
        ...,
        min_length=1,
        max_length=LONGITUD_MAXIMA_MENSAJE,
        examples=["WINNER!! Claim your FREE prize now call 09061701461"],
    )

    @field_validator("mensaje")
    @classmethod
    def no_vacio_tras_recortar(cls, valor: str) -> str:
        # min_length=1 no detecta un mensaje que es solo espacios
        # (" " tiene longitud 1 pero no es contenido real).
        if not valor.strip():
            raise ValueError("el mensaje no puede estar vacio")
        return valor


class LoteEntrada(BaseModel):
    mensajes: List[str] = Field(
        ...,
        min_length=1,
        examples=[[
            "Hey, are we still meeting at 5?",
            "WINNER!! Claim your FREE prize now call 09061701461",
        ]],
    )

    @field_validator("mensajes")
    @classmethod
    def validar_mensajes(cls, valores: List[str]) -> List[str]:
        for v in valores:
            if not v.strip():
                raise ValueError("ningun mensaje del lote puede estar vacio")
            if len(v) > LONGITUD_MAXIMA_MENSAJE:
                raise ValueError(f"un mensaje supera el maximo de {LONGITUD_MAXIMA_MENSAJE} caracteres")
        return valores


class PrediccionSalida(BaseModel):
    etiqueta: str = Field(examples=["spam"])
    probabilidad_spam: float = Field(examples=[0.97])
    version_modelo: str = Field(examples=["1"])


class LotePrediccionSalida(BaseModel):
    predicciones: List[PrediccionSalida]


class SaludSalida(BaseModel):
    estado: str = Field(examples=["ok"])
    modelo: str = Field(examples=["spam-detector"])
    version: str | None = Field(examples=["1"])
    alias: str = Field(examples=["produccion"])
    cargado_en: datetime | None


class RecargaSalida(BaseModel):
    estado: str = Field(examples=["ok"])
    version_anterior: str | None
    version_nueva: str


# ---------------------------------------------------------------
# Helpers de prediccion y logging
# ---------------------------------------------------------------
def _predecir_lote(modelo, mensajes: List[str]) -> List[PrediccionSalida]:
    # Un solo predict_proba vectorizado para todo el lote en vez de
    # un loop llamando al modelo mensaje por mensaje: mismo
    # resultado, muchas menos pasadas por el vectorizador TF-IDF.
    probabilidades = modelo.predict_proba(mensajes)
    idx_spam = list(modelo.classes_).index("spam")

    resultados = []
    for probs in probabilidades:
        prob_spam = float(probs[idx_spam])
        etiqueta = "spam" if prob_spam >= 0.5 else "ham"
        resultados.append((etiqueta, prob_spam))
    return resultados


def _registrar_predicciones(mensajes: List[str], resultados, version_modelo: str) -> None:
    """
    Guarda cada prediccion servida en produccion junto con el
    mensaje original. Este log es la materia prima para comparar
    contra los datos de entrenamiento y detectar data drift mas
    adelante (ver tests/drift.py), por lo que un fallo aqui NUNCA
    debe tumbar la respuesta al cliente.
    """
    try:
        RUTA_LOG_PREDICCIONES.parent.mkdir(parents=True, exist_ok=True)
        with open(RUTA_LOG_PREDICCIONES, "a", encoding="utf-8") as f:
            for mensaje, (etiqueta, prob_spam) in zip(mensajes, resultados):
                linea = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "mensaje": mensaje,
                    "prediccion": etiqueta,
                    "probabilidad_spam": prob_spam,
                    "version_modelo": version_modelo,
                }
                f.write(json.dumps(linea, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("No se pudo escribir el log de predicciones: %s", exc)


def _modelo_activo():
    with _candado:
        if estado.modelo is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"El modelo no esta disponible ({estado.error}). "
                    "Intenta /recargar o revisa que el alias exista en el registro."
                ),
            )
        return estado.modelo, estado.version


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------
@app.get("/salud", response_model=SaludSalida)
def salud():
    with _candado:
        if estado.modelo is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"El modelo no esta disponible: {estado.error}",
            )
        return SaludSalida(
            estado="ok",
            modelo=NOMBRE_MODELO,
            version=estado.version,
            alias=ALIAS_MODELO,
            cargado_en=estado.cargado_en,
        )


@app.post("/predecir", response_model=PrediccionSalida)
def predecir(entrada: MensajeEntrada):
    modelo, version = _modelo_activo()
    (etiqueta, prob_spam), = _predecir_lote(modelo, [entrada.mensaje])
    _registrar_predicciones([entrada.mensaje], [(etiqueta, prob_spam)], version)
    return PrediccionSalida(etiqueta=etiqueta, probabilidad_spam=prob_spam, version_modelo=version)


@app.post("/predecir-lote", response_model=LotePrediccionSalida)
def predecir_lote(entrada: LoteEntrada):
    modelo, version = _modelo_activo()
    resultados = _predecir_lote(modelo, entrada.mensajes)
    _registrar_predicciones(entrada.mensajes, resultados, version)
    return LotePrediccionSalida(
        predicciones=[
            PrediccionSalida(etiqueta=etiqueta, probabilidad_spam=prob_spam, version_modelo=version)
            for etiqueta, prob_spam in resultados
        ]
    )


@app.post("/recargar", response_model=RecargaSalida)
def recargar():
    version_anterior = estado.version
    try:
        modelo, version_nueva, cargado_en = cargar_modelo()
    except Exception as exc:
        # Si la recarga falla, el modelo previamente cargado sigue
        # sirviendo: no se pisa `estado` hasta tener el reemplazo listo.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No se pudo recargar el modelo desde el registro: {exc}",
        )

    with _candado:
        estado.modelo = modelo
        estado.version = version_nueva
        estado.cargado_en = cargado_en
        estado.error = None

    logger.info("Modelo recargado: %s -> %s", version_anterior, version_nueva)
    return RecargaSalida(estado="ok", version_anterior=version_anterior, version_nueva=version_nueva)
