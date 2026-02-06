import pandas as pd

df1 = pd.read_csv("resenasNegativasParte1.csv")
df2 = pd.read_csv("resenasNegativasParte2.csv")
df3 = pd.read_csv("resenasNegativasParte3.csv")
df4 = pd.read_csv("resenasNegativasParte4.csv")
df5 = pd.read_csv("resenasNegativasParte5.csv")
df6 = pd.read_csv("resenasNegativasParte6.csv")

df7 = pd.read_csv("airbnbResenas1.csv")
df8 = pd.read_csv("airbnbResenas2.csv")

df9 = pd.read_csv("4_estrellas.csv")
df10 = pd.read_csv("5_estrellas_2.csv")
df11 = pd.read_csv("5_estrellas_3.csv")

# Unificar todos
df_total = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

df_total.head()

print(len(df_total))

df_airbnb = pd.concat([df7, df8], ignore_index=True)
df_airbnb["columna_2"] = None
df_airbnb = df_airbnb[["columna_1", "columna_2"]]

df_airbnb.head()

print(len(df_airbnb))

df_estrellas = pd.concat([df9, df10, df11], ignore_index=True)

print(len(df_estrellas))

df_final = pd.concat([df_total, df_airbnb, df_estrellas], ignore_index=True)

df_final.head()


print(len(df_final))




from sentiment_analysis_spanish import sentiment_analysis
import re

sentiment = sentiment_analysis.SentimentAnalysisSpanish()

palabras_positivas = []
palabras_neutras = []
palabras_negativas = []

for texto in df_final["columna_1"]:
    
    texto = str(texto).lower()
    
    # separar palabras
    palabras = re.findall(r'\b\w+\b', texto)
    
    for palabra in palabras:
        score = sentiment.sentiment(palabra)  # devuelve valor entre 0 y 1
        
        if score > 0.6:
            palabras_positivas.append(palabra)
        elif score < 0.4:
            palabras_negativas.append(palabra)
        else:
            palabras_neutras.append(palabra)

print("Positivas:", palabras_positivas[:20])
print("Neutras:", palabras_neutras[:20])
print("Negativas:", palabras_negativas[:20])

print(len(palabras_positivas))
print(len(palabras_neutras))
print(len(palabras_negativas))


