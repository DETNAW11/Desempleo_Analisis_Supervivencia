# Modelo de Duración de Desempleo: Cox vs. Deep Learning (DeepSurv)

Análisis de supervivencia aplicado al tiempo de búsqueda de empleo en población rural colombiana, comparando un modelo estadístico clásico (Cox de riesgos proporcionales) contra una red neuronal (DeepSurv) entrenada con la misma función de pérdida.

*[English version](README_EN.md)*

## El problema

¿Cuánto tiempo tarda una persona en conseguir empleo, y qué factores sociodemográficos aceleran o retrasan ese proceso? En lugar de tratar esto como un problema de clasificación binaria (empleado/desempleado), se modela como un problema de **análisis de supervivencia**: el tiempo hasta que ocurre un evento (conseguir empleo), permitiendo capturar tanto si el evento ocurrió como cuánto tardó en ocurrir.

## Los datos

Microdatos de la **Gran Encuesta Integrada de Hogares (GEIH)** del DANE, filtrados a población rural (`clase == 2`). Fuente pública de descarga: [microdatos.dane.gov.co](https://microdatos.dane.gov.co/index.php/catalog/central/search?q=GEIH)

Tras la limpieza (filtrado de población ocupada, eliminación de "No sabe/No responde", y remoción de valores faltantes), la muestra final quedó en **356 observaciones**, particionadas en 284 de entrenamiento y 72 de prueba.

**Variables utilizadas:** sexo, edad (agrupada), nivel educativo, jefe de hogar, estado civil, estrato (agrupado), canal de búsqueda de empleo, ingreso no laboral, y tiempo de búsqueda en semanas.

**Limitación importante y honesta:** el subconjunto final de datos no contiene observaciones censuradas — todas las personas en la muestra final efectivamente consiguieron empleo en algún punto. Esto significa que el análisis compara *velocidad* de consecución de empleo entre distintos perfiles, no la probabilidad de nunca conseguirlo. Con un tamaño de muestra de 356, los resultados deben interpretarse con cautela, especialmente para categorías con pocos casos (por ejemplo, se fusionó "Postgrado" con "Universitario" en Nivel Educativo por baja representación).

## Metodología

### 1. Modelo de Cox (referencia clásica)

Modelo de riesgos proporcionales de Cox, con regularización (`penalizer=0.1`) para estabilidad numérica dado el tamaño de muestra reducido. Implementado con [`lifelines`](https://lifelines.readthedocs.io/).

### 2. DeepSurv (red neuronal)

Red neuronal simple (una capa oculta de 8 neuronas, dropout 0.3) entrenada con la **misma función de pérdida** que el Cox clásico: la log-verosimilitud parcial (aproximación de Breslow). Esto permite una comparación matemáticamente justa entre ambos enfoques. Implementado en PyTorch puro, con regularización L2 (`weight_decay`) para mitigar sobreajuste dado el tamaño de muestra.

La arquitectura se mantuvo deliberadamente simple: con solo 284 observaciones de entrenamiento, una red más profunda tendría un riesgo alto de sobreajuste sin ninguna ventaja real.

## Resultados

| Modelo | C-index (test) |
|---|---|
| Cox (clásico) | **0.6056** |
| Red Neuronal (DeepSurv) | 0.5970 |

![Comparación de C-index](comparacion_c_index.png)

El modelo de Cox clásico obtuvo un desempeño ligeramente superior al de la red neuronal. La diferencia es pequeña, pero consistente con lo esperable dado el tamaño de la muestra y la naturaleza de las variables (sociodemográficas, con relaciones mayormente lineales o de bajo orden con el riesgo).

### Variables significativas (Cox, p < 0.05)

- **Ingreso no laboral** (p < 0.005): tener una fuente de ingreso no laboral reduce la velocidad de consecución de empleo — coherente con menor urgencia inmediata de emplearse.
- **Edad 41-60 años** (p < 0.005): este grupo etario tarda significativamente más en conseguir empleo que los menores de 28 años.
- **Estado civil (viudos)** (p = 0.03): consiguen empleo más rápido que el grupo base, aunque con un intervalo de confianza amplio dado el bajo número de casos en esta categoría.

### Curvas de supervivencia estimadas

![Curvas de supervivencia](curvas_supervivencia_cox.png)

Comparación entre un perfil de bajo riesgo y uno de alto riesgo según el modelo de Cox, mostrando la probabilidad de permanecer desempleado a lo largo del tiempo.

## Conclusión: ¿cuándo usar cada enfoque?

Este proyecto no busca argumentar que el deep learning "no sirve" para análisis de supervivencia en general — el hallazgo es más específico y más útil: **con un dataset pequeño (cientos de observaciones) y relaciones predominantemente lineales entre variables sociodemográficas y el riesgo, un modelo de Cox regularizado iguala o supera a una red neuronal**, con la ventaja adicional de ofrecer coeficientes interpretables y significancia estadística directa.

Se esperaría que una red neuronal aporte valor real en escenarios con:
- Datasets sustancialmente más grandes (miles o decenas de miles de observaciones).
- Relaciones no lineales fuertes o interacciones complejas entre variables que un modelo lineal no captura sin especificación manual.
- Variables de alta dimensionalidad (texto, imágenes, series temporales densas) donde el aprendizaje de representaciones es una ventaja estructural, no solo una opción.

## Estructura del repositorio

```
├── data_processing.py    # Limpieza y preparación de los datos (Python, migrado desde R)
├── cox_model.py          # Modelo de Cox de riesgos proporcionales
├── deepsurv_model.py     # Red neuronal (DeepSurv) en PyTorch
├── comparacion_c_index.png
├── curvas_supervivencia_cox.png
└── README.md
```

## Tecnologías

Python · pandas · lifelines · PyTorch · scikit-learn · matplotlib

## Origen

Este análisis parte de un modelo original desarrollado en R (`survival`, `coxph`) durante una consultoría estadística, migrado y extendido a Python con la incorporación del componente de deep learning para esta comparación.