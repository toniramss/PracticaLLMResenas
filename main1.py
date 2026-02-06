import pandas as pd
import re
import unicodedata

df7 = pd.read_csv("pruebaResenyas1.csv")

df8 = pd.read_csv("airbnbResenas2.csv")



# Convertimos columnas a sets (quitando NaN y espacios)
pos_set = set(df7["Positivas"].dropna().astype(str).str.lower().str.strip())
neu_set = set(df7["Neutras"].dropna().astype(str).str.lower().str.strip())
neg_set = set(df7["Negativas"].dropna().astype(str).str.lower().str.strip())


def normalizar(texto: str) -> str:
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto

def tokenizar(texto: str):
    texto = normalizar(texto)
    return re.findall(r"[a-zñ]+", texto)

def clasificar_resena(texto: str):
    tokens = tokenizar(texto)

    score = 0
    for t in tokens:
        if t in pos_set:
            score += 1
        elif t in neg_set:
            score -= 1

    if score > 0:
        return "positiva", score
    elif score < 0:
        return "negativa", score
    else:
        return "neutra", score
    
resenasPositivas = 0
resenasNeutras = 0
resenasNegativas = 0

for resena in df8["columna_1"].dropna().astype(str):
    clasificacion, puntaje = clasificar_resena(resena)
    print(f"Reseña: {resena}\nClasificación: {clasificacion}, Puntaje: {puntaje}\n")

    if clasificacion == "positiva":
        resenasPositivas += 1
    elif clasificacion == "negativa":
        resenasNegativas += 1
    else:
        resenasNeutras += 1


print(f"Total de reseñas positivas: {resenasPositivas}")
print(f"Total de reseñas neutras: {resenasNeutras}")
print(f"Total de reseñas negativas: {resenasNegativas}")

