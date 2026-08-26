import os
import sys
import time
import subprocess
import requests
import pandas as pd
from datetime import datetime

# Definir colores para imprimir más lindo en consola
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def pausa(mensaje):
    print(f"\n{Colors.WARNING}>>> {mensaje} (Presiona ENTER para continuar){Colors.ENDC}")
    if not os.environ.get("AUTO_DEMO"):
        input()
    else:
        time.sleep(2)
    print()

def ejecutar_comando(comando):
    print(f"{Colors.OKCYAN}$ {comando}{Colors.ENDC}")
    # En Windows, subprocess llama al entorno actual, pero sys.executable asegura que usemos nuestro python local
    result = subprocess.run(comando, shell=True)
    return result.returncode

def main():
    API_URL = "http://localhost:8000"
    
    print(f"{Colors.HEADER}{Colors.BOLD}="*60)
    print("DEMOSTRACIÓN END-TO-END DE MLOps LIFECYCLE")
    print("="*60 + f"{Colors.ENDC}")
    
    pausa("Paso 1: Verificando estado inicial de la API (/salud)")
    try:
        res = requests.get(f"{API_URL}/salud")
        print(f"Respuesta de la API:\n{json.dumps(res.json(), indent=2) if res.ok else res.text}")
    except Exception as e:
        print(f"{Colors.FAIL}La API no parece estar corriendo. Por favor ejecuta 'uvicorn src.api:app' en otra terminal.{Colors.ENDC}")
        sys.exit(1)
        
    pausa("Paso 2: Enviar los 244 mensajes modernos a /predecir-lote")
    df_moderno = pd.read_csv("data/nuevos/spam_moderno.csv")
    mensajes = df_moderno["mensaje"].tolist()
    verdaderos = df_moderno["etiqueta"].tolist()
    
    payload = {"mensajes": mensajes}
    res = requests.post(f"{API_URL}/predecir-lote", json=payload)
    predicciones = res.json().get("predicciones", [])
    
    pausa("Paso 3: Mostrar cuántos se colaron (recall real sobre spam moderno)")
    # Calcular metricas
    falsos_negativos = 0
    total_spam = 0
    for v, p in zip(verdaderos, predicciones):
        if v == "spam":
            total_spam += 1
            if p["etiqueta"] != "spam":
                falsos_negativos += 1
                
    recall = (total_spam - falsos_negativos) / total_spam
    print(f"Total mensajes spam moderno: {total_spam}")
    print(f"Spam detectado correctamente: {total_spam - falsos_negativos}")
    print(f"Spam que se COLÓ (falsos negativos): {falsos_negativos}")
    print(f"{Colors.FAIL}Recall de la versión actual sobre spam moderno: {recall:.3f}{Colors.ENDC}")
    
    pausa("Paso 4: Correr monitorear.py -> reporte de drift")
    ejecutar_comando(f'"{sys.executable}" src/monitorear.py')
    
    pausa("Paso 5: Correr reentrenar.py -> v2 registrada")
    ejecutar_comando(f'"{sys.executable}" src/reentrenar.py')
    
    pausa("Paso 6: Llamar a /recargar en la API")
    res = requests.post(f"{API_URL}/recargar")
    print(f"Respuesta: {res.json() if res.ok else res.text}")
    
    pausa("Paso 7: Reenviar los mismos mensajes -> mostrar la mejora")
    res = requests.post(f"{API_URL}/predecir-lote", json=payload)
    predicciones = res.json().get("predicciones", [])
    falsos_negativos_v2 = 0
    for v, p in zip(verdaderos, predicciones):
        if v == "spam":
            if p["etiqueta"] != "spam":
                falsos_negativos_v2 += 1
                
    recall_v2 = (total_spam - falsos_negativos_v2) / total_spam
    print(f"Spam que se COLÓ con la v2: {falsos_negativos_v2}")
    print(f"{Colors.OKGREEN}Recall de la v2 sobre spam moderno: {recall_v2:.3f}{Colors.ENDC}")
    
    pausa("Paso 8: Rollback a v1 y demostrar que vuelve a fallar")
    # Para hacer rollback a la versión 1 explícita:
    ejecutar_comando(f'"{sys.executable}" src/rollback.py 1')
    
    # Recargar la API
    print(f"\nRecargando la API...")
    requests.post(f"{API_URL}/recargar")
    
    # Reenviar
    res = requests.post(f"{API_URL}/predecir-lote", json=payload)
    predicciones = res.json().get("predicciones", [])
    falsos_negativos_v1 = 0
    for v, p in zip(verdaderos, predicciones):
        if v == "spam":
            if p["etiqueta"] != "spam":
                falsos_negativos_v1 += 1
                
    recall_v1 = (total_spam - falsos_negativos_v1) / total_spam
    print(f"Spam que se COLÓ tras rollback: {falsos_negativos_v1}")
    print(f"{Colors.FAIL}Recall tras rollback a v1: {recall_v1:.3f}{Colors.ENDC}")
    
    print(f"{Colors.OKGREEN}{Colors.BOLD}¡Demostración completada con éxito!{Colors.ENDC}")

if __name__ == "__main__":
    # Necesitamos importar json aquí
    import json
    main()
