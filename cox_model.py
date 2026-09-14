import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split

basef = pd.read_csv("desempleo_limpio.csv")

# Fusionar Postgrado (5) con Universitario (4) por bajo número de casos
basef["Nivel_Educativo"] = basef["Nivel_Educativo"].replace(5.0, 4.0)

categoricas = ["Sexo", "Edad", "Nivel_Educativo", "Jefe_Hogar",
               "Estado_Civil", "Estrato", "Canal_Busqueda"]

basef_dummies = pd.get_dummies(basef, columns=categoricas, drop_first=True)

train, test = train_test_split(basef_dummies, test_size=0.2, random_state=42)

# Verificación de seguridad: eliminar cualquier columna dummy con varianza 0 en train
cols_dummy = [c for c in train.columns if c not in ["Tiempo_Busqueda", "Ocupados"]]
cols_sin_varianza = [c for c in cols_dummy if train[c].std() == 0]
if cols_sin_varianza:
    print("Columnas eliminadas por varianza 0:", cols_sin_varianza)
    train = train.drop(columns=cols_sin_varianza)
    test = test.drop(columns=cols_sin_varianza)

# Penalizer para dar más estabilidad con una muestra pequeña
cph = CoxPHFitter(penalizer=0.1)
cph.fit(train, duration_col="Tiempo_Busqueda", event_col="Ocupados")

cph.print_summary()

c_index_test = cph.score(test, scoring_method="concordance_index")
print("C-index en test:", c_index_test)

train.to_csv("train_desempleo.csv", index=False)
test.to_csv("test_desempleo.csv", index=False)