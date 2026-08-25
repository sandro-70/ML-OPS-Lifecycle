import pandas as pd

df = pd.read_csv("./data/raw/spam.csv", encoding = "latin-1")

df = df[["v1","v2"]]
df.columns = ["tipo", "mensaje"]
print("Antes de limpiar", df.shape)
df = df.drop_duplicates()
df = df.dropna()
print("Despues de limpiar: ",df.shape)

print(df["tipo"].value_counts())
print(df["tipo"].value_counts(normalize=True).mul(100).round(2))

df.to_csv("data/raw/spam_limpio.csv", index=False)