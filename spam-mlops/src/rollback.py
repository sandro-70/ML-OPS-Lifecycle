import sys
import mlflow
from mlflow.tracking import MlflowClient

# Configuracion
mlflow.set_tracking_uri("sqlite:///mlflow.db")
MODEL_NAME = "spam-detector"
ALIAS_PROD = "produccion"

def listar_versiones():
    client = MlflowClient()
    print("="*60)
    print(f"VERSIONES DISPONIBLES PARA '{MODEL_NAME}'")
    print("="*60)
    try:
        versiones = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versiones:
            print("No hay versiones registradas.")
            return

        for v in versiones:
            # Obtener las métricas del run asociado a la versión si existe
            run = client.get_run(v.run_id)
            metrics = run.data.metrics
            
            # Obtener alias de la version
            aliases = v.aliases
            alias_str = f" [Alias: {', '.join(aliases)}]" if aliases else ""
            
            # Formatear salida
            metricas_str = ", ".join([f"{k}: {val:.4f}" for k, val in metrics.items()])
            if not metricas_str:
                metricas_str = "No hay métricas registradas en este Run"
                
            print(f"Versión {v.version}{alias_str}")
            print(f"  Run ID : {v.run_id}")
            print(f"  Status : {v.status}")
            print(f"  Métricas: {metricas_str}\n")
            
    except Exception as e:
        print(f"Error al obtener versiones: {e}")

def aplicar_rollback(version):
    client = MlflowClient()
    try:
        client.set_registered_model_alias(MODEL_NAME, ALIAS_PROD, str(version))
        print(f"[OK] ÉXITO: El alias '{ALIAS_PROD}' ha sido asignado a la versión {version}.")
    except Exception as e:
        print(f"[ERROR] ERROR al asignar el alias: {e}")

def main():
    args = sys.argv[1:]
    
    if len(args) == 0:
        listar_versiones()
        print("\nUso para rollback: python src/rollback.py <numero_de_version>")
    elif len(args) == 1:
        version = args[0]
        print(f"Iniciando rollback de la versión {version}...")
        aplicar_rollback(version)
    else:
        print("Uso incorrecto. Argumentos excedentes.")
        print("Uso: python src/rollback.py [numero_de_version]")
        sys.exit(1)

if __name__ == "__main__":
    main()
