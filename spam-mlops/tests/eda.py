import pandas as pd

RUTA = "./data/raw/spam_limpio.csv"


def separador(titulo):
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


df = pd.read_csv(RUTA, encoding="latin-1")


separador("1. FORMA GENERAL DEL ARCHIVO")
print(f"Filas:    {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")
print(f"\nNombres de columnas: {list(df.columns)}")
print("\nTipos de dato:")
print(df.dtypes.to_string())


separador("2. PRIMERAS FILAS")
pd.set_option("display.max_colwidth", 70)
print(df.head(5).to_string())


separador("3. VALORES VACIOS (NaN) POR COLUMNA")
vacios = pd.DataFrame({
    "no_nulos": df.notna().sum(),
    "nulos": df.isna().sum(),
    "pct_nulos": (df.isna().sum() / len(df) * 100).round(2),
})
print(vacios.to_string())
print("\nLectura: una columna con casi 100% de nulos es basura del archivo,")
print("no un dato faltante que debas rellenar.")


separador("4. COLUMNAS UTILES vs COLUMNAS BASURA")
umbral = 0.5
utiles = [c for c in df.columns if df[c].notna().mean() > umbral]
basura = [c for c in df.columns if df[c].notna().mean() <= umbral]
print(f"Columnas con mas del {umbral:.0%} de datos (utiles): {utiles}")
print(f"Columnas casi vacias (candidatas a eliminar):        {basura}")

for col in basura:
    n = df[col].notna().sum()
    if n:
        print(f"\n  Ejemplo de contenido en '{col}' ({n} filas con dato):")
        print(f"    {df[col].dropna().iloc[0][:90]}")


separador("5. LA COLUMNA DE ETIQUETAS")
col_etiqueta = utiles[0]
print(f"Columna analizada: '{col_etiqueta}'")
print(f"Valores unicos: {df[col_etiqueta].nunique()}")
print("\nConteo:")
print(df[col_etiqueta].value_counts().to_string())
print("\nPorcentaje:")
print(df[col_etiqueta].value_counts(normalize=True).mul(100).round(2).to_string())

mayoritaria = df[col_etiqueta].value_counts(normalize=True).max() * 100
print(f"\nLA CLASE MAYORITARIA REPRESENTA {mayoritaria:.1f}% DEL TOTAL.")
print(f"Un modelo que siempre respondiera la clase mayoritaria")
print(f"acertaria {mayoritaria:.1f}% de las veces sin aprender nada.")
print("Por eso la exactitud (accuracy) sola no sirve como metrica aqui.")


separador("6. DUPLICADOS")
col_texto = utiles[1]
dup_totales = df.duplicated(subset=utiles).sum()
dup_texto = df.duplicated(subset=[col_texto]).sum()
print(f"Filas identicas en etiqueta + mensaje: {dup_totales}")
print(f"Mensajes repetidos (sin importar etiqueta): {dup_texto}")
print(f"Filas unicas que quedarian: {len(df) - dup_totales}")

if dup_totales:
    print("\nMensajes mas repetidos:")
    top = df[col_texto].value_counts().head(3)
    for texto, veces in top.items():
        print(f"  [{veces} veces] {str(texto)[:65]}")

print("\nPor que importa: si un mensaje duplicado cae en entrenamiento Y en")
print("prueba, el modelo lo 'reconoce' y tu precision sale inflada.")


separador("7. CONFLICTOS DE ETIQUETADO")
conflictos = df.groupby(col_texto)[col_etiqueta].nunique()
conflictos = conflictos[conflictos > 1]
print(f"Mensajes identicos con etiquetas contradictorias: {len(conflictos)}")
if len(conflictos):
    print("Ejemplos:")
    for texto in conflictos.index[:3]:
        print(f"  {str(texto)[:65]}")
else:
    print("Ninguno. El etiquetado es consistente.")


separador("8. LONGITUD DE LOS MENSAJES")
df["_largo"] = df[col_texto].astype(str).str.len()
print("Estadisticas generales:")
print(df["_largo"].describe().round(1).to_string())
print(f"\nPromedio de caracteres por clase:")
print(df.groupby(col_etiqueta)["_largo"].mean().round(1).to_string())
print("\nPista: si una clase es notablemente mas larga, la longitud")
print("por si sola ya es una senal predictiva.")


separador("9. EJEMPLOS DE CADA CLASE")
for clase in df[col_etiqueta].dropna().unique()[:2]:
    print(f"\n--- {str(clase).upper()} ---")
    muestra = df[df[col_etiqueta] == clase][col_texto].head(4)
    for i, texto in enumerate(muestra, 1):
        print(f"{i}. {str(texto)[:110]}")


separador("10. PALABRAS MAS FRECUENTES POR CLASE")
for clase in df[col_etiqueta].dropna().unique()[:2]:
    textos = df[df[col_etiqueta] == clase][col_texto].astype(str).str.lower()
    palabras = textos.str.split().explode()
    palabras = palabras[palabras.str.len() > 3]
    print(f"\n{str(clase).upper()}: ", end="")
    print(", ".join(palabras.value_counts().head(10).index))

print("\nEstas son las palabras que TF-IDF va a aprender a pesar")
print("en la fase 2. Compara las dos listas: ahi esta la senal.")


separador("FIN DE LA EXPLORACION")
print("Nada fue modificado. El archivo original sigue intacto.")