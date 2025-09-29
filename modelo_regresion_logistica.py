# modelo_regresion_logistica.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Cargar dataset
df = pd.read_csv("dataset_abandono_colegio.csv")
print("Primeras filas del dataset:")
print(df.head())

print("\nInformación del dataset:")
print(df.info())

print("\nValores nulos por columna:")
print(df.isnull().sum())

# 2. Limpiar la variable objetivo
# Reemplazar valores vacíos o no válidos por NaN
df["Abandono"] = df["Abandono"].replace(["", " ", "NA", "NaN"], np.nan)

# Eliminar filas con NaN en la variable objetivo
df = df.dropna(subset=["Abandono"])

# Si la variable es texto tipo "Sí"/"No", convertir a binario
df["Abandono"] = df["Abandono"].map({"Sí": 1, "No": 0})

# Verificar que no queden nulos
print("\n¿Quedan nulos en Abandono?:", df["Abandono"].isnull().sum())

# 3. Definir X e y
X = df.drop("Abandono", axis=1)
y = df["Abandono"]

# 4. Identificar variables numéricas y categóricas
numeric_features = ["Edad", "Promedio_Colegio", "Examen_Admision", "Promedio_Primer_Semestre"]
categorical_features = ["Genero", "Origen", "Nivel_Socioeconomico", "Beca", "Prestamo"]

# 5. Pipelines de preprocesamiento
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 6. Definir modelo
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 7. Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Entrenar modelo
model.fit(X_train, y_train)

# 9. Evaluar modelo
y_pred = model.predict(X_test)

print("\n--- Resultados del modelo ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))


