# Índice de Salud Vegetal

## Datos y definición del índice

El conjunto de Kaggle `ziya07/plant-health-data` es **sintético**. El archivo original contiene 1.200 mediciones de 10 plantas. `Plant_Health_Status` se generó mediante una regla fija sobre humedad del suelo y nitrógeno. Las otras nueve variables se generaron como ruido independiente respecto de esa etiqueta.

El índice de 0 a 100 es **una regla definida por el proyecto**. Cada variable se convierte en un puntaje trapezoidal de 0 a 1. La fórmula operativa es `85 × min(puntaje de humedad, puntaje de nitrógeno)` más los aportes ponderados de las otras nueve variables, que suman hasta 15 puntos. El diagnóstico informa cuál de las dos variables tiene el menor puntaje, muestra ambas en caso de empate y lista las variables fuera de su rango ideal.

`indice_salud.formula` en [config.yaml](../config.yaml) permite elegir `minimo` o `suma`. La fórmula anterior de suma pondera humedad con 68, nitrógeno con 17 y las nueve restantes con 15 en conjunto. Con el mínimo, la puntuación global promedio pasó de 71,73 a 62,37 (−9,36 puntos). El cambio fue de −17,62 en `High Stress`, −6,06 en `Moderate Stress` y 0,00 en `Healthy`.

## Calibración y concordancia

Se reservaron las plantas 6 y 7 para la prueba (240 filas, 20 %). Las otras ocho plantas formaron el conjunto de calibración y entrenamiento (960 filas, 80 %). Se usó `StratifiedGroupKFold` con cinco particiones y semilla 42, agrupando por `Plant_ID` y estratificando por `Plant_Health_Status`. Se eligió el fold más cercano al 20 % global y por estado a partir de tamaños y conteos de etiquetas. Los conteos por estado fueron 399/321/240 en calibración y 101/80/59 en prueba, en orden de mayor a menor estrés.

Los umbrales se **calibraron contra la etiqueta sintética**. En el 80 % se probaron 15 combinaciones (Crítico por debajo de 40, 50 o 60 y Excelente desde 75, 80, 85, 90 o 95). Se maximizó la proporción de aciertos con el recall promedio por clase como desempate. Después se aplicaron los cortes elegidos al 20 %. La clase intermedia es Atención. La correspondencia usada fue Crítico con `High Stress`, Atención con `Moderate Stress` y Excelente con `Healthy`.

| Fórmula | Crítico < | Excelente ≥ | Calibración, 960 filas | Prueba, 240 filas | Total, 1.200 filas |
|---|---:|---:|---:|---:|---:|
| Suma | 60 | 95 | 799/960 (83,23 %) | 196/240 (81,67 %) | 995/1.200 (82,92 %) |
| Mínimo | 50 | 90 | 881/960 (91,77 %) | 220/240 (91,67 %) | 1.101/1.200 (91,75 %) |

| Recall por estado en prueba | High Stress | Moderate Stress | Healthy |
|---|---:|---:|---:|
| Suma | 84,16 % | 90,00 % | 66,10 % |
| Mínimo | 92,08 % | 85,00 % | 100,00 % |

Los cortes operativos del mínimo son **Crítico < 50**, **Atención ≥ 50 y < 90**, y **Excelente ≥ 90**. La tabla cruzada sobre el total muestra los 99 desacuerdos. Hay 38 muestras `High Stress` en Atención y 61 `Moderate Stress` en Excelente.

| Categoría del índice | High Stress | Moderate Stress | Healthy | Total |
|---|---:|---:|---:|---:|
| Crítico | 462 | 0 | 0 | 462 |
| Atención | 38 | 340 | 0 | 378 |
| Excelente | 0 | 61 | 299 | 360 |
| **Total** | **500** | **401** | **299** | **1.200** |

El control previo aprobó antes de entrenar. Las categorías superaron el mínimo de 50 muestras y sus promedios por estado subieron estrictamente (32,15 en `High Stress`, 74,93 en `Moderate Stress` y 96,05 en `Healthy`). [Histograma del índice](figures/16_indice_v2_distribucion.png).

## Regresión del índice

El entrenamiento principal usó nueve entradas ambientales y de suelo. Se excluyeron `Chlorophyll_Content`, `Electrochemical_Signal`, la fecha y sus derivados, el identificador de planta y la etiqueta de estado. El objetivo fue el índice determinista con la fórmula del mínimo. Se usó validación cruzada `GroupKFold` de cinco folds en el 80 % y se evaluó en las dos plantas reservadas. MAE y RMSE se expresan en puntos del índice.

| Modelo | R² medio CV | MAE medio CV | RMSE medio CV | R² prueba | MAE prueba | RMSE prueba |
|---|---:|---:|---:|---:|---:|---:|
| Media de referencia | −0,0039 | 26,1084 | 29,0310 | −0,0001 | 24,6495 | 27,4932 |
| Lineal | 0,7250 | 11,3570 | 15,1268 | 0,7264 | 10,4856 | 14,3811 |
| Ridge | 0,7250 | 11,3591 | 15,1271 | 0,7265 | 10,4839 | 14,3780 |
| Polinomial de grado 2 | 0,8903 | 7,6342 | 9,5805 | 0,9031 | 6,8598 | 8,5576 |
| Random Forest Regressor | 0,9954 | 1,4417 | 1,9416 | 0,9956 | 1,3941 | 1,8216 |

### Experimento de 2 frente a 11 variables

Se conservaron el objetivo, la semilla, los folds y las plantas de prueba. Dos entradas corresponden a humedad y nitrógeno. Once entradas corresponden a todos los componentes del índice. Los resultados de la tabla son sobre el 20 % reservado.

| Modelo | R² (2) | R² (11) | MAE (2) | MAE (11) | RMSE (2) | RMSE (11) |
|---|---:|---:|---:|---:|---:|---:|
| Media de referencia | −0,0001 | −0,0001 | 24,6495 | 24,6495 | 27,4932 | 27,4932 |
| Lineal | 0,7277 | 0,7266 | 10,3704 | 10,4623 | 14,3471 | 14,3757 |
| Ridge | 0,7278 | 0,7267 | 10,3679 | 10,4610 | 14,3442 | 14,3726 |
| Polinomial de grado 2 | 0,9022 | 0,9025 | 6,8619 | 6,8350 | 8,5996 | 8,5851 |
| Random Forest Regressor | 0,9944 | 0,9959 | 1,6352 | 1,3376 | 2,0531 | 1,7582 |

Con 11 entradas, el Random Forest redujo su MAE en 0,2976 puntos frente al modelo con dos entradas. Las nueve variables de ruido contribuyen por definición al índice que la regresión busca reproducir. [Gráfico de la comparación](figures/21_comparacion_11vs2_vars.png).

## Limitaciones

- La concordancia de categorías mide el acuerdo con una etiqueta sintética. R², MAE y RMSE miden la aproximación a una fórmula construida a partir de las mismas variables de sensor. **Ninguna de estas métricas mide capacidad predictiva real de salud vegetal.**
- La fórmula del mínimo y la grilla se habían explorado sobre las 1.200 filas. El procedimiento seleccionó los cortes solo con el 80 %, pero el 20 % procede del mismo conjunto ya explorado. La cifra total incluye las filas de calibración. La prueba comprende dos plantas sintéticas y no constituye validación externa.
- Los rangos, sus unidades y las ponderaciones son supuestos del proyecto. Harían falta mediciones y resultados de salud de plantas reales, con contexto de especie, cultivo y sensor, para validarlos.

### Rangos pendientes de fuente

| Variable | Unidad en la configuración | Rango ideal | Rango absoluto |
|---|---|---:|---:|
| `Soil_pH` | pH | 6,0–7,0 | 5,5–7,5 |
| `Ambient_Temperature` | °C | 20–28 | 18–30 |
| `Potassium_Level` | mg K/kg de suelo | 25–42 | 10–50 |
| `Light_Intensity` | lux, relativo al sensor | 500–900 | 200–1.000 |
| `Phosphorus_Level` | mg P/kg de suelo | 25–42 | 10–50 |
| `Chlorophyll_Content` | unidades relativas, SPAD aprox. | 38–50 | 20–50,5 |
| `Humidity` | % | 50–65 | 40–70 |
| `Soil_Temperature` | °C | 17–23 | 15–25 |
| `Electrochemical_Signal` | mV, relativo al sensor | 0,8–1,5 | 0–2 |

## Reproducción

```powershell
python -X utf8 src/calibracion_umbrales.py
python -X utf8 src/train_regression_v2.py --solo-control
python -X utf8 src/train_regression_v2.py
python -X utf8 src/experimento_variables.py
python -m pytest -q
```
