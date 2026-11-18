"""
train_regression_v2.py
======================
Regresión del índice con entradas ambientales y de suelo.

Modelos: Baseline | Lineal | Ridge | Polinomial grado-2 | RF Regressor
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feature_engineering as fe
from calibracion_umbrales import (
    CATEGORIAS, ESTADOS, ESTADO_COL, GROUP_COL, N_FOLDS, SEED, TARGET, dividir_por_planta,
)
from diagnostico import cargar_config, categorizar, umbrales_categoria

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(BASE, "reports", "figures")
os.makedirs(FIGS, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

MIN_MUESTRAS_CATEGORIA = 50

FEATURES = [
    "Soil_Moisture", "Ambient_Temperature", "Soil_Temperature",
    "Humidity", "Light_Intensity", "Soil_pH",
    "Nitrogen_Level", "Phosphorus_Level", "Potassium_Level",
]


# ── Pipelines ──────────────────────────────────────────────────────────────────
def make_pipelines():
    return {
        "Baseline (media)"  : Pipeline([("sc", StandardScaler()),
                                         ("m", DummyRegressor(strategy="mean"))]),
        "Lineal"            : Pipeline([("sc", StandardScaler()),
                                         ("m", LinearRegression())]),
        "Ridge (a=1)"       : Pipeline([("sc", StandardScaler()),
                                         ("m", Ridge(alpha=1.0))]),
        "Polinomial grado-2": Pipeline([("sc", StandardScaler()),
                                         ("poly", PolynomialFeatures(degree=2,
                                                                      include_bias=False)),
                                         ("m", LinearRegression())]),
        "RF Regressor"      : Pipeline([("sc", StandardScaler()),
                                         ("m", RandomForestRegressor(
                                             n_estimators=200,
                                             max_depth=None,
                                             min_samples_leaf=3,
                                             random_state=SEED,
                                             n_jobs=-1))]),
    }


# ── CV con GroupKFold ─────────────────────────────────────────────────────────
def cv_grouped(pipelines, X, y, groups, label=""):
    gkf = GroupKFold(n_splits=N_FOLDS)
    rows = []
    for nombre, pipe in pipelines.items():
        res = cross_validate(
            pipe, X, y, cv=gkf, groups=groups,
            scoring=["r2", "neg_mean_absolute_error",
                     "neg_root_mean_squared_error"],
            return_train_score=True,
            error_score="raise",
        )
        rows.append({
            "Experimento"    : label,
            "Modelo"         : nombre,
            "Train R2"       : res["train_r2"].mean(),
            "CV R2 (media)"  : res["test_r2"].mean(),
            "CV R2 (std)"    : res["test_r2"].std(),
            "CV MAE (media)" : -res["test_neg_mean_absolute_error"].mean(),
            "CV RMSE (media)": -res["test_neg_root_mean_squared_error"].mean(),
        })
    return pd.DataFrame(rows)


# ── Metricas en test set ───────────────────────────────────────────────────────
def metricas_test(pipelines, X_tr, y_tr, X_te, y_te, label=""):
    rows = []
    preds = {}
    for nombre, pipe in pipelines.items():
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        preds[nombre] = y_pred
        rows.append({
            "Experimento": label,
            "Modelo"     : nombre,
            "R2"         : r2_score(y_te, y_pred),
            "MAE"        : mean_absolute_error(y_te, y_pred),
            "RMSE"       : np.sqrt(mean_squared_error(y_te, y_pred)),
        })
    return pd.DataFrame(rows), preds


# ── Helpers de impresion ───────────────────────────────────────────────────────
def print_df(df, titulo):
    print(f"\n{'─'*72}")
    print(titulo)
    print("─" * 72)
    fmt = {c: "{:.4f}".format for c in df.select_dtypes("number").columns}
    print(df.to_string(index=False, formatters=fmt))


def control_previo(df, umbrales):
    """Muestra los diagnósticos y detiene la ejecución si el índice falla."""
    indice = df[TARGET]
    atencion, excelente = umbrales
    categorias = indice.map(lambda valor: categorizar(valor, atencion, excelente))
    categorias = categorias.rename("Categoría del índice")
    conteos = categorias.value_counts().reindex(CATEGORIAS, fill_value=0)
    medias = df.groupby(ESTADO_COL)[TARGET].mean().reindex(ESTADOS)
    tabla = pd.crosstab(categorias, df[ESTADO_COL]).reindex(
        index=CATEGORIAS, columns=ESTADOS, fill_value=0
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(indice, bins=40, range=(0, 100), color="#347A61")
    ax.axvline(atencion, color="#A12721", ls="--",
               label=f"Umbral de Atención ({atencion:g})")
    ax.axvline(excelente, color="#A36200", ls="--",
               label=f"Umbral de Excelente ({excelente:g})")
    ax.set(xlim=(0, 100), xlabel="Índice de Salud Vegetal", ylabel="Muestras",
           title="Distribución del Índice de Salud Vegetal")
    ax.legend()
    ruta_histograma = os.path.join(FIGS, "16_indice_v2_distribucion.png")
    fig.savefig(ruta_histograma, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nHistograma guardado en {ruta_histograma}")
    print("\nMuestras por categoría del índice")
    print(conteos.to_string())
    print("\nÍndice promedio por Plant_Health_Status")
    print(medias.to_string(float_format="{:.4f}".format))
    print("\nTabla cruzada de categoría del índice por Plant_Health_Status")
    print(tabla.to_string(), flush=True)

    errores = []
    insuficientes = conteos[conteos < MIN_MUESTRAS_CATEGORIA]
    if not insuficientes.empty:
        detalle = ", ".join(f"{nombre} ({n})" for nombre, n in insuficientes.items())
        errores.append(f"Categorías con menos de {MIN_MUESTRAS_CATEGORIA} muestras: {detalle}.")

    high, moderate, healthy = medias
    if not high < moderate < healthy:
        errores.append("Los promedios deben cumplir High Stress < Moderate Stress < Healthy.")

    if errores:
        raise SystemExit("Entrenamiento cancelado. " + " ".join(errores))
    print("\nControl previo aprobado.")


def preparar_dataset():
    config = cargar_config(fe.CONFIG)
    df = fe.regenerar_dataset(config)
    control_previo(df, umbrales_categoria(config))
    return df


def evaluar_modelos(train, test, features, etiqueta):
    modelos = make_pipelines()
    x_train, x_test = train[features], test[features]
    y_train, y_test = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    cv = cv_grouped(modelos, x_train, y_train, train[GROUP_COL], etiqueta)
    metricas, predicciones = metricas_test(modelos, x_train, y_train, x_test, y_test, etiqueta)
    print_df(cv, f"CV (GroupKFold 5-fold) — {etiqueta}")
    print_df(metricas, f"Prueba por planta — {etiqueta}")
    return cv, metricas, modelos, predicciones


# ── Main ───────────────────────────────────────────────────────────────────────
def main(*, solo_control=False):
    print("=" * 72)
    print("REGRESION v2 — indice_salud  |  GroupKFold por Plant_ID")
    print("=" * 72)

    # ── 1. Datos ───────────────────────────────────────────────────
    df = preparar_dataset()
    if solo_control:
        return

    df_tr, df_te = dividir_por_planta(df)
    print(f"\nTrain: {len(df_tr)} filas (plantas {sorted(df_tr[GROUP_COL].unique())})")
    print(f"Test : {len(df_te)} filas (plantas {sorted(df_te[GROUP_COL].unique())})")

    df_cv, df_test, modelos, predicciones = evaluar_modelos(
        df_tr, df_te, FEATURES, "9 variables ambientales y de suelo"
    )
    y_te = df_te[TARGET].to_numpy()

    # ── 6. Coeficientes ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("COEFICIENTES — Regresion Lineal (9 variables, espacio escalado)")
    print("=" * 72)
    lin_fitted = modelos["Lineal"]
    coef = lin_fitted.named_steps["m"].coef_
    intercept = lin_fitted.named_steps["m"].intercept_
    df_coef = pd.DataFrame({
        "Feature": FEATURES,
        "Coef"   : coef,
        "Abs"    : np.abs(coef),
    }).sort_values("Abs", ascending=False)
    df_coef["Coef"] = df_coef["Coef"].map("{:+.5f}".format)
    print(f"Intercepto: {intercept:.4f}")
    print(df_coef[["Feature", "Coef"]].to_string(index=False))

    print("\n" + "=" * 72)
    print("IMPORTANCIA — RF Regressor (9 variables)")
    print("=" * 72)
    rf_fitted = modelos["RF Regressor"]
    imp_rf = pd.Series(
        rf_fitted.named_steps["m"].feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=False)
    print(imp_rf.map("{:.5f}".format).to_string())
    acum = 0
    for feat, v in imp_rf.items():
        acum += v
        print(f"  {feat:<27} {v:.4f}  acum={acum:.4f}")

    # ── 7. Graficos ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("GRAFICOS")
    print("=" * 72)

    # 17. Predicho vs Real — 5 modelos (9 variables)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    colores = ["#9E9E9E", "#2196F3", "#9C27B0", "#FF9800", "#4CAF50"]
    for ax, (nombre, y_pred), color in zip(axes, predicciones.items(), colores):
        r2  = r2_score(y_te, y_pred)
        ax.scatter(y_te, y_pred, s=18, alpha=0.45, color=color, edgecolors="none")
        lim = (min(y_te.min(), y_pred.min()) - 2, max(y_te.max(), y_pred.max()) + 2)
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Real"); ax.set_ylabel("Predicho")
        ax.set_title(f"{nombre}\nR²={r2:.3f}", fontsize=8, fontweight="bold")
    plt.suptitle("Predicho vs Real — 9 variables", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(FIGS, "17_predicho_vs_real_all_models.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"  -> {p}")

    # 18. Residuos del modelo con mayor R² medio en validación cruzada
    best_name = df_cv.loc[df_cv["CV R2 (media)"].idxmax(), "Modelo"]
    y_pred_best = predicciones[best_name]
    residuos = y_te - y_pred_best

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # residuos vs predicho
    axes[0, 0].scatter(y_pred_best, residuos, s=18, alpha=0.45,
                        color="#4CAF50", edgecolors="none")
    axes[0, 0].axhline(0, color="red", lw=1.5, ls="--")
    std_r = residuos.std()
    axes[0, 0].axhline( 2*std_r, color="orange", lw=1, ls=":")
    axes[0, 0].axhline(-2*std_r, color="orange", lw=1, ls=":", label="+/-2std")
    axes[0, 0].set_xlabel("Predicho"); axes[0, 0].set_ylabel("Residuo")
    axes[0, 0].set_title("Residuos vs Predichos", fontweight="bold")
    axes[0, 0].legend(fontsize=8)
    # histograma
    sns.histplot(residuos, kde=True, bins=30, ax=axes[0, 1], color="#4CAF50")
    axes[0, 1].set_title("Distribucion de Residuos", fontweight="bold")
    # QQ-plot
    (osm, osr), (sl, ic, r_qq) = stats.probplot(residuos, dist="norm")
    axes[1, 0].scatter(osm, osr, s=15, alpha=0.5, color="#4CAF50")
    xline = np.array([osm[0], osm[-1]])
    axes[1, 0].plot(xline, sl*xline + ic, "r--", lw=1.5)
    axes[1, 0].set_xlabel("Cuantiles Teoricos"); axes[1, 0].set_ylabel("Observados")
    axes[1, 0].set_title(f"QQ-Plot (r={r_qq:.4f})", fontweight="bold")
    # residuos en orden
    axes[1, 1].plot(residuos, lw=0.6, alpha=0.7, color="#4CAF50")
    axes[1, 1].axhline(0, color="red", lw=1.2, ls="--")
    axes[1, 1].set_xlabel("Muestra"); axes[1, 1].set_ylabel("Residuo")
    axes[1, 1].set_title("Residuos en Orden", fontweight="bold")
    plt.suptitle(f"Analisis de Residuos — {best_name} (9 variables)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(FIGS, "18_residuos_mejor_modelo.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"  -> {p}")

    # 19. Coeficientes Lineal vs Ridge (9 variables)
    ridge_fitted = modelos["Ridge (a=1)"]
    coef_lin = modelos["Lineal"].named_steps["m"].coef_
    coef_rid = ridge_fitted.named_steps["m"].coef_
    x_pos = np.arange(len(FEATURES))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x_pos - 0.2, coef_lin, 0.38, label="Lineal", color="#2196F3", alpha=0.85)
    ax.bar(x_pos + 0.2, coef_rid, 0.38, label="Ridge",  color="#E91E63", alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(FEATURES, rotation=35, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Coeficiente (espacio escalado)")
    ax.set_title("Coeficientes — Lineal vs Ridge (9 variables)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    p = os.path.join(FIGS, "19_coeficientes_lineal_ridge.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"  -> {p}")

    # 20. RF Regressor — importancia de variables
    fig, ax = plt.subplots(figsize=(9, 5))
    imp_sorted = imp_rf.sort_values()
    bars = ax.barh(imp_sorted.index, imp_sorted.values, color="#4CAF50", alpha=0.85)
    ax.bar_label(bars, fmt="{:.4f}", padding=3, fontsize=8)
    ax.set_xlabel("Importancia (reducción de impureza)")
    ax.set_title("RF Regressor — Importancia de Variables (9 variables)", fontweight="bold")
    plt.tight_layout()
    p = os.path.join(FIGS, "20_rf_importancia_regressor.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"  -> {p}")

    print("\nTodos los graficos guardados en reports/figures/")
    print("=" * 72)

    # Devuelve tablas para informe
    return df_cv, df_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regresión del Índice de Salud Vegetal.")
    parser.add_argument("--solo-control", action="store_true",
                        help="Regenera el índice y muestra el control previo sin entrenar.")
    main(solo_control=parser.parse_args().solo_control)
