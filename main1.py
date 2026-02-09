import pandas as pd
import re
import unicodedata

df7 = pd.read_csv("pruebaResenyas1.csv")


datasetEjemploResenas = pd.read_csv("airbnbResenas2.csv")

datasetEjemploResenasNegativas = pd.read_csv("resenasNegativasParte1.csv")


datasetPositivas = pd.read_csv("positivas.csv")
datasetNeutras = pd.read_csv("neutras.csv")
datasetNegativas = pd.read_csv("negativas.csv")

datasetPositivas = set(
    datasetPositivas["palabra"].dropna().astype(str).str.lower().str.strip().drop_duplicates() 
)

datasetNeutras = set(
    datasetNeutras["palabra"].dropna().astype(str).str.lower().str.strip().drop_duplicates() 
)

datasetNegativas = set(
    datasetNegativas["palabra"].dropna().astype(str).str.lower().str.strip().drop_duplicates() 
)


# Convertimos columnas a sets (quitando NaN y espacios)
#positive_set = set(df7["Positivas"].dropna().astype(str).str.lower().str.strip())
#neutral_set = set(df7["Neutras"].dropna().astype(str).str.lower().str.strip())
#negative_set = set(df7["Negativas"].dropna().astype(str).str.lower().str.strip())


def normalizar(texto: str) -> str:
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto) # Descomponer caracteres acentuados
    texto = "".join(c for c in texto if not unicodedata.combining(c)) # Eliminar marcas de acentuación
    return texto

def tokenizar(texto: str):
    texto = normalizar(texto)
    return re.findall(r"[a-zñ]+", texto)

def clasificar_resena(texto: str):
    tokens = tokenizar(texto)

    positivas = 0
    negativas = 0
    neutras = 0

    for token in tokens:
        if token in datasetPositivas:
            positivas += 1
            print(f"Token '{token}' clasificado como POSITIVO")
        elif token in datasetNegativas:
            negativas += 1
            print(f"Token '{token}' clasificado como NEGATIVO")
        else:
            neutras += 1
            print(f"Token '{token}' clasificado como NEUTRO")

    # Decisión según cantidades
    if positivas > negativas:
        clasificacion = "positiva"
    elif negativas > positivas:
        clasificacion = "negativa"
    else:
        clasificacion = "neutra"

    return clasificacion, positivas, negativas, neutras


resenasPositivas = 0
resenasNeutras = 0
resenasNegativas = 0

for resena in datasetEjemploResenasNegativas["columna_2"].dropna().astype(str):

    clasificacion, n_pos, n_neg, n_neu = clasificar_resena(resena)

    print(f"Reseña: {resena}")
    print(f"Palabras positivas: {n_pos} | Palabras negativas: {n_neg} | Palabras neutras: {n_neu}")
    print(f"Clasificación: {clasificacion}\n")

    if clasificacion == "positiva":
        resenasPositivas += 1
    elif clasificacion == "negativa":
        resenasNegativas += 1
    else:
        resenasNeutras += 1


print(f"Total de reseñas positivas: {resenasPositivas}")
print(f"Total de reseñas neutras: {resenasNeutras}")
print(f"Total de reseñas negativas: {resenasNegativas}")
