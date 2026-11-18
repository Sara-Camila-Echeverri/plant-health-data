"""Calibración y evaluación de umbrales con plantas separadas."""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from diagnostico import cargar_config, categorizar, umbrales_categoria
from feature_engineering import DATA, calcular_indices

SEED = 42
N_FOLDS = 5
TARGET = "indice_salud"
GROUP_COL = "Plant_ID"
ESTADO_COL = "Plant_Health_Status"
CATEGORIAS = ("Critico", "Atencion Requerida", "Excelente")
ESTADOS = ("High Stress", "Moderate Stress", "Healthy")
ESTADO_A_CATEGORIA = dict(zip(ESTADOS, CATEGORIAS))
UMBRALES_CRITICO = (40, 50, 60)
UMBRALES_EXCELENTE = (75, 80, 85, 90, 95)


def dividir_por_planta(df):
    """Escoge el fold más cercano al 20 % global y al 20 % de cada estado."""
    objetivo = 1 / N_FOLDS
    conteos = df[ESTADO_COL].value_counts()
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    def distancia(particion):
        _, prueba = particion
        conteos_prueba = df.iloc[prueba][ESTADO_COL].value_counts()
        fracciones = conteos_prueba.reindex(conteos.index, fill_value=0) / conteos
        return abs(len(prueba) / len(df) - objetivo) + (fracciones - objetivo).abs().mean()

    particiones = splitter.split(df, df[ESTADO_COL], groups=df[GROUP_COL])
    entrenamiento, prueba = min(particiones, key=distancia)
    return df.iloc[entrenamiento], df.iloc[prueba]


def concordancia(df, umbral_atencion, umbral_excelente):
    categorias = df[TARGET].map(
        lambda indice: categorizar(indice, umbral_atencion, umbral_excelente)
    )
    correctos = categorias.eq(df[ESTADO_COL].map(ESTADO_A_CATEGORIA))
    recalls = correctos.groupby(df[ESTADO_COL]).mean().reindex(ESTADOS)
    return int(correctos.sum()), float(correctos.mean()), recalls


def calibrar(df_entrenamiento):
    candidatos = []
    for atencion in UMBRALES_CRITICO:
        for excelente in UMBRALES_EXCELENTE:
            correctos, _, recalls = concordancia(df_entrenamiento, atencion, excelente)
            candidatos.append((atencion, excelente, correctos, recalls.mean()))
    atencion, excelente, _, _ = max(candidatos, key=lambda fila: (fila[2], fila[3]))
    return atencion, excelente


def main():
    config = cargar_config()
    datos = pd.read_csv(DATA)
    train, test = dividir_por_planta(datos)
    for nombre, parte in (("80 %", train), ("20 %", test)):
        print(f"{nombre}: {len(parte)} muestras, plantas {sorted(parte[GROUP_COL].unique())}")
        print(parte[ESTADO_COL].value_counts().reindex(ESTADOS).to_string())

    for formula in ("suma", "minimo"):
        config_formula = {**config, "formula": formula}
        train_formula = train.assign(indice_salud=calcular_indices(train, config_formula))
        cortes = calibrar(train_formula)
        test_formula = test.assign(indice_salud=calcular_indices(test, config_formula))
        total = pd.concat([train_formula, test_formula])

        comparaciones = {"calibrados": cortes}
        if formula == config["formula"] and cortes != umbrales_categoria(config):
            comparaciones["operativos"] = umbrales_categoria(config)
        for tipo, umbrales in comparaciones.items():
            print(f"\n{formula}, cortes {tipo}: {umbrales[0]:g}/{umbrales[1]:g}")
            for nombre, parte in (("80 %", train_formula), ("20 %", test_formula), ("Total", total)):
                correctos, accuracy, recalls = concordancia(parte, *umbrales)
                recall_texto = ", ".join(f"{estado}={valor:.2%}" for estado, valor in recalls.items())
                print(f"{nombre}: {correctos}/{len(parte)} = {accuracy:.2%} | {recall_texto}")


if __name__ == "__main__":
    main()
