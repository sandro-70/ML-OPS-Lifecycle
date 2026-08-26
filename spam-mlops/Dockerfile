# El modelo NO se hornea dentro de la imagen: se carga en runtime
# desde el registro de MLflow (models:/spam-detector@produccion), tal
# como lo hace src/api.py. Esto significa que mlflow.db y mlruns/
# deben montarse como volumenes al correr el contenedor (ver notas
# al final de este archivo); la imagen por si sola no sirve
# predicciones sin ese volumen.
FROM python:3.11-slim

WORKDIR /app

# Copiar solo requirements.txt primero: mientras el archivo no
# cambie, Docker reutiliza esta capa de cache y no reinstala todo el
# arbol de dependencias (mlflow y scikit-learn son pesados) cada vez
# que cambia el codigo de src/.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# Usuario sin privilegios: si el proceso de la API se ve
# comprometido, no hereda automaticamente permisos de root dentro
# del contenedor.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Se consulta /salud en vez de solo verificar que el proceso este
# vivo: si el modelo no cargo (falta el volumen del registro, o el
# alias no existe), /salud responde 503 y el contenedor debe
# marcarse unhealthy aunque uvicorn siga respondiendo.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/salud')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# -------------------------------------------------------------------
# Como montar el registro de MLflow al correr el contenedor
#
# api.py carga el modelo con tracking URI "sqlite:///mlflow.db" (ruta
# relativa al cwd del proceso, que aqui es /app por el WORKDIR). Para
# que el contenedor vea el mismo registro que tienes en local:
#
#   docker run -p 8000:8000 \
#     -v "C:\Develop\Projects\ML-OPS Lifecycle\spam-mlops\mlflow.db:/app/mlflow.db" \
#     -v "C:\Develop\Projects\ML-OPS Lifecycle\spam-mlops\mlruns:/app/mlruns" \
#     spam-detector-api
#
# Monta AMBOS: mlflow.db (los metadatos: nombre del modelo, alias,
# version) y mlruns/ (los artefactos reales: model.skops). El
# artifact_location que MLflow guarda en mlflow.db es una ruta
# RELATIVA ("file:mlruns/1/..."), asi que se resuelve contra el cwd
# del proceso (/app) y coincide con este segundo volumen sin
# configuracion adicional. Si mlflow.db se monta solo, sin mlruns/,
# el modelo "carga" (los metadatos existen) pero falla al leer el
# archivo del modelo real.
# -------------------------------------------------------------------
