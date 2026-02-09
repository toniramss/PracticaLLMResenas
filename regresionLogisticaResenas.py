import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


listaResenasNegativas = pd.read_csv("resenasNegativasParte1.csv")["columna_2"].dropna().astype(str).tolist()
listaResenasPositivas = pd.read_csv("airbnbResenas2.csv")["columna_1"].dropna().astype(str).tolist()

print("Total reseñas negativas:", len(listaResenasNegativas))
print("Total reseñas positivas:", len(listaResenasPositivas))

df_negativas = pd.DataFrame({
    "resena": listaResenasNegativas,
    "sentimiento": "negativa"
})

df_positivas = pd.DataFrame({
    "resena": listaResenasPositivas,
    "sentimiento": "positiva"
})

dataset_resenas = pd.concat(
    [df_positivas, df_negativas],
    ignore_index=True
)

# Mezclar el dataset
dataset_resenas = dataset_resenas.sample(frac=1, random_state=42).reset_index(drop=True)


print(dataset_resenas.head())
print("Total reseñas:", len(dataset_resenas))







df = dataset_resenas.dropna(subset=["resena", "sentimiento"]).copy()
df["resena"] = df["resena"].astype(str).str.lower().str.strip()

X = df["resena"]
y = df["sentimiento"]

# 1) split ANTES
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2) fit SOLO en train
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95)
X_train = vectorizer.fit_transform(X_train_text)
X_test  = vectorizer.transform(X_test_text)

modelo = LogisticRegression(max_iter=100, class_weight="balanced")
modelo.fit(X_train, y_train)

print("Clases:", modelo.classes_)
print("Intercept:", modelo.intercept_)


y_pred = modelo.predict(X_test)
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte:\n", classification_report(y_test, y_pred))



def predecir(texto):
    x = vectorizer.transform([texto.lower()])
    return modelo.predict(x)[0], modelo.predict_proba(x).max()

tests = [
    "La habitación era cómoda pero el ruido fue molesto",
    "Todo correcto, sin más",
    "No volvería, muy mala experiencia",
    "El anfitrión fue amable y el piso estaba limpio"
]

for t in tests:
    print(t, "->", predecir(t))
