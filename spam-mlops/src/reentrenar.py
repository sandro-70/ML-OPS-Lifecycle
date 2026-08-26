import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Configuracion de MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("spam-mlops")

RUTA_RAW = "data/raw/spam_limpio.csv"
RUTA_NUEVOS = "data/nuevos/spam_moderno.csv"
MODEL_NAME = "spam-detector"
ALIAS_PROD = "produccion"
F1_MINIMO = 0.99  # calidad minima absoluta (clase "spam") para poder desplegar

def cargar_datos():
    df_raw = pd.read_csv(RUTA_RAW)
    df_nuevos = pd.read_csv(RUTA_NUEVOS)

    # data/raw/spam_limpio.csv (dataset v2, versionado con DVC) ya
    # trae los mensajes modernos anexados, pero quedaron en la
    # columna "etiqueta" en vez de "tipo" porque el anexado no
    # unifico nombres de columna: los 5169 mensajes clasicos solo
    # tienen "tipo", los 244 modernos solo tienen "etiqueta". Sin
    # este fillna, dropna(subset=['etiqueta']) descartaba los 5169
    # clasicos y el "reentrenamiento" terminaba entrenando (y
    # evaluando) solo con los mensajes modernos duplicados dos veces
    # (una vez desde aqui, otra desde RUTA_NUEVOS) -de ahi el F1
    # perfecto que no significaba nada.
    df_raw["tipo"] = df_raw["tipo"].fillna(df_raw["etiqueta"])
    df_raw = df_raw[["tipo", "mensaje"]].rename(columns={"tipo": "etiqueta"})

    df = pd.concat([df_raw, df_nuevos], ignore_index=True)
    # Los 244 modernos aparecen en ambas fuentes (ya anexados en
    # RUTA_RAW y otra vez en RUTA_NUEVOS): sin deduplicar por
    # mensaje, quedarian contados doble.
    df = df.drop_duplicates(subset=["mensaje"])
    df.dropna(subset=['mensaje', 'etiqueta'], inplace=True)
    return train_test_split(df["mensaje"], df["etiqueta"], test_size=0.2, random_state=42, stratify=df["etiqueta"])

def main():
    print("="*60)
    print("REENTRENAMIENTO CONDICIONAL")
    print("="*60)
    
    # 1. Cargar dataset actualizado
    print("Cargando y dividiendo datos actualizados...")
    X_train, X_test, y_train, y_test = cargar_datos()
    print(f"Total entrenamiento: {len(X_train)} | Total prueba: {len(X_test)}")

    # 2. Obtener modelo actual en produccion
    client = MlflowClient()
    try:
        model_prod = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS_PROD}")
        print(f"Modelo actual en producción cargado correctamente.")
    except Exception as e:
        print(f"No se pudo cargar el modelo de producción: {e}")
        print("Saliendo para evitar reemplazar sin comparación.")
        sys.exit(1)

    # 3. Evaluar modelo actual en el nuevo set de prueba
    y_pred_prod = model_prod.predict(X_test)
    f1_prod_spam = f1_score(y_test, y_pred_prod, pos_label="spam")
    
    # 4. Entrenar el nuevo modelo
    print("\nEntrenando nuevo modelo candidato (TF-IDF Bigramas + Naive Bayes)...")
    vectorizador = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clasificador = MultinomialNB(alpha=0.1) # No aplica class_weight
    
    modelo_candidato = Pipeline([
        ("tfidf", vectorizador),
        ("clf", clasificador),
    ])
    
    modelo_candidato.fit(X_train, y_train)
    
    # Evaluar modelo candidato
    y_pred_cand = modelo_candidato.predict(X_test)
    f1_cand_spam = f1_score(y_test, y_pred_cand, pos_label="spam")
    
    # 5. Comparar
    print("\n" + "-"*40)
    print("COMPARATIVA DE F1-SCORE (Clase 'spam')")
    print("-"*40)
    print(f"Producción actual : {f1_prod_spam:.4f}")
    print(f"Candidato nuevo   : {f1_cand_spam:.4f}")
    print(f"Mínimo requerido  : {F1_MINIMO:.4f}")
    print("-"*40)

    if f1_cand_spam > f1_prod_spam and f1_cand_spam >= F1_MINIMO:
        print("\n¡El modelo candidato supera al modelo de producción y la calidad mínima!")
        print("Registrando y promoviendo la nueva versión...")
        
        with mlflow.start_run(run_name="reentrenamiento-automatico"):
            mlflow.log_metric("f1_spam_test", f1_cand_spam)
            
            # Registrar modelo
            model_info = mlflow.sklearn.log_model(modelo_candidato, "modelo")
            model_uri = model_info.model_uri
            
            # Registrar en el Model Registry
            result = mlflow.register_model(model_uri, MODEL_NAME)
            version_nueva = result.version
            
            # Asignar alias
            client.set_registered_model_alias(MODEL_NAME, ALIAS_PROD, version_nueva)
            print(f"La versión {version_nueva} es ahora el nuevo alias '{ALIAS_PROD}'.")
    else:
        if f1_cand_spam <= f1_prod_spam:
            print("\nEl modelo candidato NO supera al modelo actual.")
        else:
            print(f"\nEl modelo candidato no alcanza la calidad mínima requerida ({F1_MINIMO:.4f}).")
        print("Se descarta el candidato. No se promueve nada.")
    
    print("="*60)

if __name__ == "__main__":
    main()
