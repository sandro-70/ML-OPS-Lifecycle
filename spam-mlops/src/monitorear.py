import json
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def calcular_longitud(df):
    if "mensaje" in df.columns:
        df["longitud"] = df["mensaje"].fillna("").apply(len)
    return df

def main():
    print("Iniciando monitoreo de data drift...")
    ruta_ref = Path("data/raw/spam_limpio.csv")
    ruta_act = Path("data/produccion/predicciones.jsonl")
    
    if not ruta_ref.exists():
        print(f"ERROR: No se encontró el dataset de referencia en {ruta_ref}")
        sys.exit(2)
        
    if not ruta_act.exists():
        print(f"No hay predicciones recientes en {ruta_act}. No se puede evaluar drift.")
        sys.exit(0)

    # Cargar referencia
    df_ref = pd.read_csv(ruta_ref)
    df_ref = calcular_longitud(df_ref)
    # Evidently necesita que la columna de prediccion se llame igual o la pasemos
    # En la referencia, tenemos 'etiqueta'. La renombramos a 'prediccion' para comparar distribuciones.
    df_ref = df_ref.rename(columns={"etiqueta": "prediccion"})

    # Cargar actual
    # Las predicciones están en JSONL
    registros = []
    with open(ruta_act, "r", encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                registros.append(json.loads(linea))
    
    df_act = pd.DataFrame(registros)
    if df_act.empty:
        print("El archivo de predicciones está vacío.")
        sys.exit(0)
        
    df_act = calcular_longitud(df_act)

    # Alinear columnas para evidently
    cols_evaluacion = ["mensaje", "longitud", "prediccion"]
    
    # Rellenar faltantes por seguridad
    for c in cols_evaluacion:
        if c not in df_act.columns:
            df_act[c] = None

    df_ref = df_ref[cols_evaluacion].copy()
    df_act = df_act[cols_evaluacion].copy()

    # Configurar Evidently Report
    # Usaremos DataDriftPreset que evalúa todas las columnas
    column_mapping = {
        "target": None,
        "prediction": "prediccion",
        "text_features": ["mensaje"],
        "numerical_features": ["longitud"],
        "categorical_features": ["prediccion"]
    }
    
    reporte = Report(metrics=[
        DataDriftPreset(stattest_threshold=0.05)
    ])

    print("Calculando métricas de drift...")
    # Pasar el mapping no siempre es nativo en report.run pero podemos pasarlo en column_mapping
    # Wait, evidently 0.4.x pass column_mapping via ColumnMapping object
    from evidently.pipeline.column_mapping import ColumnMapping
    cm = ColumnMapping(
        target=None,
        prediction="prediccion",
        text_features=["mensaje"],
        numerical_features=["longitud"],
        categorical_features=["prediccion"]
    )
    
    reporte.run(reference_data=df_ref, current_data=df_act, column_mapping=cm)

    # Guardar reporte HTML
    fecha_str = datetime.now().strftime("%Y%m%d")
    Path("reportes").mkdir(exist_ok=True)
    ruta_reporte = f"reportes/drift_{fecha_str}.html"
    reporte.save_html(ruta_reporte)
    print(f"Reporte HTML generado en: {ruta_reporte}")

    # Extraer resultado
    # DataDriftPreset devuelve dataset_drift
    resultados = reporte.as_dict()
    # Buscar el dataset_drift
    # Es un poco complejo navegar el dict de metrics. Lo más seguro es:
    drift_metrics = None
    for metric in resultados["metrics"]:
        if metric["metric"] == "DatasetDriftMetric":
            drift_metrics = metric["result"]
            break
            
    if not drift_metrics:
        print("ERROR: No se pudo extraer DatasetDriftMetric del reporte.")
        sys.exit(2)

    hay_drift = drift_metrics["dataset_drift"]
    columnas_con_drift = drift_metrics["number_of_drifted_columns"]
    total_columnas = drift_metrics["number_of_columns"]

    print("\n" + "="*50)
    if hay_drift:
        print("VEREDICTO: DRIFT DETECTADO")
    else:
        print("VEREDICTO: SIN DRIFT")
    
    print(f"Métricas:")
    print(f" - Columnas analizadas: {total_columnas}")
    print(f" - Columnas con drift: {columnas_con_drift}")
    
    # Imprimir detalles por columna
    for feature, details in drift_metrics.get("drift_by_columns", {}).items():
        drift_score = details.get("drift_score", "N/A")
        feature_drift = details.get("drift_detected", False)
        print(f"   * {feature}: drift={'Si' if feature_drift else 'No'} (p-value/score={drift_score})")
    
    print("="*50 + "\n")

    if hay_drift:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
