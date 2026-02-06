import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


df7 = pd.read_csv("pruebaResenyas1.csv")

# Descargar solo la primera vez
#nltk.download("vader_lexicon")

# Inicializar el analizador
sia = SentimentIntensityAnalyzer()

def clasificar_resena_vader(texto: str):
    """
    Clasifica un texto como positiva, negativa o neutra
    usando VADER (NLTK)
    """
    scores = sia.polarity_scores(str(texto))
    compound = scores["compound"]

    if compound >= 0.05:
        return "positiva", compound
    elif compound <= -0.05:
        return "negativa", compound
    else:
        return "neutra", compound
    

# EXTRAER LÉXICO DE VADER
# ==============================
lexicon = sia.lexicon

palabras_positivas = []
palabras_negativas = []
palabras_neutras = []

for palabra, score in lexicon.items():
    if score > 0:
        palabras_positivas.append(palabra)
    elif score < 0:
        palabras_negativas.append(palabra)
    else:
        palabras_neutras.append(palabra)

# ==============================
# RESULTADOS
# ==============================
print("Total palabras positivas:", len(palabras_positivas))
print("Total palabras negativas:", len(palabras_negativas))
print("Total palabras neutras:", len(palabras_neutras))

print("\nEjemplos palabras positivas:")
print(palabras_positivas[:30])

print("\nEjemplos palabras negativas:")
print(palabras_negativas[:30])

print("\nEjemplos palabras neutras:")
print(palabras_neutras[:30])