"""Compara 2 y 11 entradas para aproximar el mismo índice."""

from pathlib import Path

from diagnostico import VARIABLES_PRINCIPALES, cargar_config
from train_regression_v2 import (
    FIGS, dividir_por_planta, evaluar_modelos, preparar_dataset, print_df,
)
import matplotlib.pyplot as plt


def main():
    config = cargar_config()
    entradas = {"2 variables": list(VARIABLES_PRINCIPALES), "11 variables": list(config["variables"])}
    train, test = dividir_por_planta(preparar_dataset())
    resultados = {}
    for etiqueta, variables in entradas.items():
        _, metricas, _, _ = evaluar_modelos(train, test, variables, etiqueta)
        resultados[etiqueta] = metricas.set_index("Modelo")[["R2", "MAE", "RMSE"]]

    dos, once = resultados.values()
    diferencias = (once - dos).add_prefix("Delta ")
    comparacion = dos.add_suffix(" (2)").join(once.add_suffix(" (11)")).join(diferencias)
    print_df(comparacion.reset_index(), "Comparación en prueba — diferencia: 11 menos 2")

    fig, ax = plt.subplots(figsize=(11, 5))
    comparacion[["R2 (2)", "R2 (11)"]].plot.bar(
        ax=ax, color=["#2196F3", "#4CAF50"], rot=15
    )
    ax.set(xlabel="", ylabel="R² en prueba", title="Comparación de 2 y 11 variables")
    fig.tight_layout()
    ruta = Path(FIGS) / "21_comparacion_11vs2_vars.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en {ruta}")
    return comparacion


if __name__ == "__main__":
    main()
