"""
Genera mensajes de spam con vocabulario moderno para simular drift.

El dataset original (SMS Spam Collection, UCI) es de 2012: habla de
premios por SMS, tonos de celular y concursos telefonicos. El spam de
hoy habla de criptomonedas, paqueteria y verificacion de cuentas.
Ese cambio de vocabulario es exactamente el "data drift" que el
modelo entrenado en 2012 no puede manejar.

Uso: python src/generar_drift.py
Salida: data/nuevos/spam_moderno.csv
"""

import os
import random
import pandas as pd

SEMILLA = 42
N_SPAM = 200
N_HAM = 60
SALIDA = "spam-mlops/data/nuevos/spam_moderno.csv"

random.seed(SEMILLA)


# ---------------------------------------------------------------
# Plantillas de spam moderno, agrupadas por tema.
# {slots} se rellenan con las listas de abajo.
# ---------------------------------------------------------------
PLANTILLAS_SPAM = [
    # Criptomonedas
    "Your {crypto} wallet received {amount}. Verify your seed phrase at {url} to unlock",
    "URGENT: unusual login on your {exchange} account. Secure your assets now {url}",
    "{crypto} airdrop closing today. Connect wallet to claim {amount} {url}",
    "Congratulations, you qualified for our {crypto} staking bonus of {amount}. Claim {url}",
    "Final notice: your {exchange} withdrawal of {amount} is pending approval {url}",

    # Paqueteria
    "{courier}: your parcel is held at customs. Pay {small} clearance fee {url}",
    "Delivery failed for tracking {tracking}. Reschedule here {url}",
    "{courier} notice: incomplete address. Update details within 24h {url}",
    "Your package {tracking} could not be delivered. Confirm shipping fee {small} {url}",

    # Verificacion de cuentas
    "{brand} security alert: your account will be suspended. Verify identity {url}",
    "We detected a login from a new device on your {brand} account. Not you? {url}",
    "Your {brand} subscription payment failed. Update billing to avoid cancellation {url}",
    "{brand} account locked due to suspicious activity. Restore access {url}",
    "Two factor verification required for your {brand} account. Confirm at {url}",

    # Streaming y suscripciones
    "Your {streaming} membership expires today. Renew now and get {discount} off {url}",
    "{streaming} premium unlocked for you. Activate free trial {url}",
    "Payment declined for {streaming}. Update your card details {url}",

    # Trabajo remoto y ofertas
    "Remote position available: earn {amount} weekly working from home. Apply {url}",
    "Your CV was shortlisted. Complete onboarding to start earning {amount} {url}",
    "Part time data entry role, {amount} per week, no experience needed {url}",

    # Bancario y reembolsos
    "{bank} alert: a transaction of {amount} was authorized. Cancel here {url}",
    "You have an unclaimed tax refund of {amount}. Submit details {url}",
    "{bank}: your card was temporarily blocked. Reactivate at {url}",
    "Refund of {amount} approved. Confirm your account to receive funds {url}",

    # Codigos y OTP
    "Your verification code is {codigo}. Do not share. If this wasn't you visit {url}",
    "Someone requested a password reset for your {brand} account. Code {codigo} {url}",
]

# Mensajes normales modernos, para que el lote nuevo sea realista
PLANTILLAS_HAM = [
    "hey can you send me the zoom link for the meeting",
    "im running late, traffic is crazy. be there in 15",
    "did you finish the report? i can review it tonight",
    "lets grab lunch tomorrow, maybe around 1?",
    "my phone died earlier sorry, whats up",
    "can you pick up the groceries on your way back",
    "the wifi password is on the router, check the back",
    "happy birthday!! hope you have a great one",
    "sending you the photos from last weekend",
    "call me when you get a chance, nothing urgent",
    "i think i left my charger at your place",
    "the meeting got moved to thursday morning",
    "thanks for helping out yesterday, really appreciate it",
    "are we still on for the gym at 6",
    "just landed, will text you when i get to the hotel",
]

SLOTS = {
    "crypto": ["Bitcoin", "Ethereum", "USDT", "Solana", "crypto", "BTC"],
    "exchange": ["Binance", "Coinbase", "Kraken", "Crypto.com"],
    "courier": ["DHL", "FedEx", "USPS", "Royal Mail", "UPS"],
    "brand": ["PayPal", "Amazon", "Apple ID", "Microsoft", "Google", "Instagram"],
    "streaming": ["Netflix", "Spotify", "Disney Plus", "HBO Max"],
    "bank": ["HSBC", "Barclays", "Santander", "Chase"],
    "amount": ["$1250", "$890", "$3400", "0.45 BTC", "$620", "$2100", "$750"],
    "small": ["$1.99", "$2.50", "$3.20", "$0.99"],
    "discount": ["50%", "70%", "30%"],
    "url": [
        "http://secure-verify.co/x9",
        "https://bit.ly/3xKp2mQ",
        "http://account-update.net/id",
        "https://tinyurl.com/y7auth",
        "http://verify-now.io/session",
    ],
    "tracking": ["GB4471829", "US9928471", "DE7719284", "FR2284719"],
    "codigo": ["482913", "770264", "159372", "603841"],
}


def rellenar(plantilla):
    texto = plantilla
    for clave, opciones in SLOTS.items():
        marcador = "{" + clave + "}"
        while marcador in texto:
            texto = texto.replace(marcador, random.choice(opciones), 1)
    return texto


def generar():
    spam = []
    for _ in range(N_SPAM):
        plantilla = random.choice(PLANTILLAS_SPAM)
        spam.append({"etiqueta": "spam", "mensaje": rellenar(plantilla)})
    # Solo se deduplica el spam: las plantillas con slots generan
    # variantes, pero pueden repetirse por azar.
    df_spam = pd.DataFrame(spam).drop_duplicates(subset=["mensaje"])

    ham = [{"etiqueta": "ham", "mensaje": random.choice(PLANTILLAS_HAM)}
           for _ in range(N_HAM)]
    df_ham = pd.DataFrame(ham)

    df = pd.concat([df_spam, df_ham], ignore_index=True)
    return df.sample(frac=1, random_state=SEMILLA).reset_index(drop=True)


if __name__ == "__main__":
    df = generar()
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    df.to_csv(SALIDA, index=False)

    print(f"Generados {len(df)} mensajes -> {SALIDA}")
    print(df["etiqueta"].value_counts().to_string())
    print("\nEjemplos de spam moderno:")
    for m in df[df.etiqueta == "spam"]["mensaje"].head(5):
        print(f"  {m[:85]}")