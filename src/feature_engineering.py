"""
feature_engineering.py
Construye la variable objetivo 'indice_salud' (0-100) con el diagnóstico
trapezoidal y los parámetros de config.yaml.
"""

import os
from tempfile import NamedTemporaryFile

import pandas as pd

if __package__:
    from .diagnostico import cargar_config, diagnosticar
else:
    from diagnostico import cargar_config, diagnosticar


# ── rutas ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(BASE, "config.yaml")
DATA   = os.path.join(BASE, "data", "raw", "plant_health_data.csv")
OUT    = os.path.join(BASE, "data", "processed", "plant_health_processed.csv")


# ── construccion de features temporales ───────────────────────────────────────
def agregar_features_temporales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hora_dia"]   = df["Timestamp"].dt.hour
    df["dia_semana"] = df["Timestamp"].dt.dayofweek
    df["mes"]        = df["Timestamp"].dt.month
    return df


def calcular_indices(df: pd.DataFrame, config: dict) -> list[float]:
    return [diagnosticar(row, config=config).indice_salud for row in df.to_dict("records")]


def regenerar_dataset(config: dict | None = None) -> pd.DataFrame:
    """Reconstruye el CSV procesado a partir de los datos originales."""
    df = pd.read_csv(DATA, parse_dates=["Timestamp"])
    print(f"Dataset cargado: {df.shape[0]} filas")

    config = cargar_config(CONFIG) if config is None else config
    df["indice_salud"] = calcular_indices(df, config)
    print(f"indice_salud calculado — rango: [{df['indice_salud'].min():.2f}, "
          f"{df['indice_salud'].max():.2f}]  media: {df['indice_salud'].mean():.2f}")

    # features temporales
    df = agregar_features_temporales(df)

    # Reemplazar el CSV solo después de que la escritura termine.
    with NamedTemporaryFile(dir=os.path.dirname(OUT), suffix=".csv", delete=False) as temp:
        ruta_temporal = temp.name
    try:
        df.to_csv(ruta_temporal, index=False)
        os.replace(ruta_temporal, OUT)
    finally:
        if os.path.exists(ruta_temporal):
            os.unlink(ruta_temporal)
    print(f"Dataset procesado guardado en: {OUT}")
    return df


def main():
    df = regenerar_dataset()
    # validacion cruzada con Plant_Health_Status
    from scipy.stats import spearmanr
    status_map = {"Healthy": 2, "Moderate Stress": 1, "High Stress": 0}
    status_num = df["Plant_Health_Status"].map(status_map)
    rho, p = spearmanr(df["indice_salud"], status_num)
    print(f"\nValidacion — Spearman rho vs Plant_Health_Status: {rho:.4f}  (p={p:.2e})")

    medias = df.groupby("Plant_Health_Status")["indice_salud"].mean().sort_values(ascending=False)
    print("\nMedia de indice_salud por clase:")
    print(medias.to_string())

    return df


if __name__ == "__main__":
    main()
