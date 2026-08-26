"""
Fija el tracking URI de MLflow y el directorio de trabajo ANTES de
que cualquier modulo del proyecto (src.api, en particular) lo importe.

MLflow resuelve rutas relativas contra el directorio de trabajo del
PROCESO, no contra la ubicacion de este archivo, y eso aplica en DOS
lugares distintos:

1. El tracking URI en si (sqlite:///mlflow.db). Si pytest se invoca
   desde otro directorio -incluida la raiz del repo git, que tiene su
   propio mlflow.db suelto de pruebas anteriores- terminaria leyendo
   un registro vacio o equivocado.

2. La ubicacion de los ARTEFACTOS dentro de ese mismo mlflow.db. Para
   que el registro funcione montado en Docker (ver Dockerfile), la
   ubicacion de cada modelo se guarda como ruta relativa
   ("file:mlruns/1/...") en vez de absoluta. MLflow resuelve esa
   ruta relativa contra el cwd del proceso en el momento de cargar el
   modelo -no contra donde vive mlflow.db-, asi que si pytest corre
   con otro cwd (por ejemplo, desde la raiz del repo) esa resolucion
   apunta al lugar equivocado aunque el tracking URI si sea correcto.
   Por eso no basta con fijar el tracking URI: hay que fijar tambien
   el cwd.

Como pytest importa conftest.py antes de recolectar cualquier
archivo de prueba, este bloque corre primero siempre, sin depender
de que ningun test lo invoque explicitamente.
"""

import os
from pathlib import Path

import mlflow

RAIZ_PROYECTO = Path(__file__).resolve().parent

# Sin esto, cualquier prueba que dependa de la resolucion de
# artefactos relativa (ver punto 2 arriba) queda a merced de desde
# donde se haya invocado "pytest", que no es algo que el proyecto
# controle.
os.chdir(RAIZ_PROYECTO)


def _resolver_tracking_uri():
    # Variable de entorno explicita: gana siempre (permite apuntar a
    # otro registro sin tocar codigo, por ejemplo en un CI que si
    # tuviera un mlflow.db propio).
    externo = os.environ.get("MLFLOW_TRACKING_URI")
    if externo:
        return externo

    ruta_sqlite = RAIZ_PROYECTO / "mlflow.db"
    if ruta_sqlite.exists():
        return f"sqlite:///{ruta_sqlite.as_posix()}"

    ruta_mlruns = RAIZ_PROYECTO / "mlruns"
    if ruta_mlruns.exists():
        return ruta_mlruns.resolve().as_uri()

    # Ni mlflow.db ni mlruns/: entorno sin registro local (tipico de
    # CI, donde ambos estan en .gitignore). Se deja sin fijar; las
    # pruebas marcadas requiere_modelo fallarian aqui, pero esas se
    # excluyen explicitamente del comando de CI con -m "not requiere_modelo".
    return None


_uri_resuelto = _resolver_tracking_uri()
if _uri_resuelto:
    os.environ["MLFLOW_TRACKING_URI"] = _uri_resuelto
    mlflow.set_tracking_uri(_uri_resuelto)
