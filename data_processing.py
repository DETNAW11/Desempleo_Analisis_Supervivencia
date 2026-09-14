import pandas as pd
import numpy as np

# --- Cargar la base de datos ---
df = pd.read_stata("TOTAL_NACIONAL_ANUAL_2021.dta", convert_categoricals=False)

# --- Filtrar por población rural (clase == 2) ---
# Corrección: 'clase' se lee como texto ('1'/'2'), no como número
base = df[df["clase"] == "2"].copy()

# --- Seleccionar variables para el modelo ---
base1 = pd.DataFrame({
    "Tiempo_Busqueda": base["p7250"],
    "Sexo": base["p6020"],
    "Edad": base["p6040"],
    "Nivel_Educativo": base["p6220"],
    "Jefe_Hogar": base["p6050"],
    "Estado_Civil": base["p6070"],
    "Estrato": base["p4030s1a1"],
    "Canal_Busqueda": base["p6290"],
    "Ingreso_No_Laboral": base["p7510s7"],
})

# --- Crear la variable Ocupados (misma lógica que en R) ---
ocupados = base["ocupado"].fillna(0)
nocupados = base["desocupado"].apply(lambda x: 2 if x == 1 else 0).fillna(0)
ocupados = ocupados.apply(lambda x: 2 if x == 1 else 0)
base1["Ocupados"] = ocupados + nocupados

# --- Limpieza (equivalente a basef en R) ---
basef = base1.copy()

basef = basef[basef["Ocupados"] != 0].copy()
basef["Ocupados"] = basef["Ocupados"].apply(lambda x: 1 if x == 2 else 0)

basef = basef[basef["Estrato"] != 9]
basef = basef[basef["Nivel_Educativo"] != 9]
basef = basef[basef["Ingreso_No_Laboral"] != 9]

basef["Jefe_Hogar"] = basef["Jefe_Hogar"].apply(lambda x: 1 if x == 1 else 0).astype("category")
basef["Sexo"] = basef["Sexo"].apply(lambda x: 0 if x == 2 else 1).astype("category")
basef["Nivel_Educativo"] = basef["Nivel_Educativo"].astype("category")

def recodificar_canal(x):
    if x == 1:
        return 1
    elif x == 2:
        return 2
    elif x in [3, 4, 5]:
        return 3
    else:
        return np.nan

basef["Canal_Busqueda"] = basef["Canal_Busqueda"].apply(recodificar_canal).astype("category")
basef["Estado_Civil"] = basef["Estado_Civil"].astype("category")

def recodificar_estrato(x):
    if x in [1, 2]:
        return 1
    elif x in [3, 4]:
        return 2
    else:
        return 3

basef["Estrato"] = basef["Estrato"].apply(recodificar_estrato).astype("category")

def recodificar_edad(x):
    if x <= 28:
        return 1
    elif x <= 40:
        return 2
    elif x <= 60:
        return 3
    else:
        return 4

basef["Edad"] = basef["Edad"].apply(recodificar_edad).astype("category")

basef = basef.dropna()

print("Filas finales:", basef.shape)
print(basef.head())

basef.to_csv("desempleo_limpio.csv", index=False)