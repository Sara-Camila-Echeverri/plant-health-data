"""
train_regression.py
Entrena y compara tres modelos de regresion sobre indice_salud:
  - Baseline (prediccion de la media)
  - Regresion Lineal Multiple
  - Ridge

Reporta R2, MAE, RMSE, coeficientes, graficos predicho vs real
y analisis de residuos. Guarda figuras en reports/figures/.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# UTF-8 en consola Windows
if hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

# importar construccion del indice
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feature_engineering as fe

# ── configuracion ─────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS  = os.path.join(BASE, "reports", "figures")
os.makedirs(FIGS, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)

FEATURES = [
    "Soil_Moisture", "Ambient_Temperature", "Soil_Temperature",
    "Humidity", "Light_Intensity", "Soil_pH",
    "hora_dia", "dia_semana", "mes",
]
TARGET   = "indice_salud"
SEED     = 42
N_FOLDS  = 5


# ── helpers ───────────────────────────────────────────────────────────────────
def metricas(y_true, y_pred, nombre=""):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"Modelo": nombre, "R2": r2, "MAE": mae, "RMSE": rmse}


def cv_metricas(pipeline, X, y, nombre=""):
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    res = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
        return_train_score=False,
    )
    return {
        "Modelo"      : nombre,
        "CV R2 media" : res["test_r2"].mean(),
        "CV R2 std"   : res["test_r2"].std(),
        "CV MAE media": -res["test_neg_mean_absolute_error"].mean(),
        "CV MAE std"  : res["test_neg_mean_absolute_error"].std(),
        "CV RMSE media": -res["test_neg_root_mean_squared_error"].mean(),
        "CV RMSE std"  : res["test_neg_root_mean_squared_error"].std(),
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("ENTRENAMIENTO DE MODELOS DE REGRESION")
    print("=" * 65)

    # ── 1. datos ──────────────────────────────────────────────────
    processed = os.path.join(BASE, "data", "processed", "plant_health_processed.csv")
    if not os.path.exists(processed):
        print("Construyendo indice_salud...")
        df = fe.main()
    else:
        df = pd.read_csv(processed, parse_dates=["Timestamp"])
        print(f"Dataset procesado cargado: {df.shape[0]} filas")

    X = df[FEATURES]
    y = df[TARGET]

    print(f"\nFeatures ({len(FEATURES)}): {FEATURES}")
    print(f"Target  : {TARGET}  |  rango [{y.min():.2f}, {y.max():.2f}]  media={y.mean():.2f}")

    # ── 2. split 80/20 ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )
    print(f"\nTrain: {len(X_train)} muestras  |  Test: {len(X_test)} muestras")

    # ── 3. pipelines ──────────────────────────────────────────────
    baseline  = Pipeline([("scaler", StandardScaler()),
                          ("model", DummyRegressor(strategy="mean"))])
    lin_reg   = Pipeline([("scaler", StandardScaler()),
                          ("model", LinearRegression())])
    ridge     = Pipeline([("scaler", StandardScaler()),
                          ("model", Ridge(alpha=1.0))])

    modelos = {
        "Baseline (media)": baseline,
        "Regresion Lineal" : lin_reg,
        "Ridge (alpha=1)"  : ridge,
    }

    # ── 4. entrenamiento y metricas test ──────────────────────────
    print("\n" + "─" * 65)
    print("A) METRICAS EN TEST SET (80/20)")
    print("─" * 65)

    resultados_test = []
    predicciones    = {}

    for nombre, pipe in modelos.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        predicciones[nombre] = y_pred
        resultados_test.append(metricas(y_test, y_pred, nombre))

    df_test = pd.DataFrame(resultados_test)
    df_test["R2"]   = df_test["R2"].map("{:.4f}".format)
    df_test["MAE"]  = df_test["MAE"].map("{:.4f}".format)
    df_test["RMSE"] = df_test["RMSE"].map("{:.4f}".format)
    print(df_test.to_string(index=False))

    # ── 5. validacion cruzada 5-fold ──────────────────────────────
    print("\n" + "─" * 65)
    print(f"B) VALIDACION CRUZADA ({N_FOLDS}-FOLD, sobre train set)")
    print("─" * 65)

    resultados_cv = []
    for nombre, pipe in modelos.items():
        resultados_cv.append(cv_metricas(pipe, X_train, y_train, nombre))

    df_cv = pd.DataFrame(resultados_cv)
    for col in df_cv.columns[1:]:
        df_cv[col] = df_cv[col].map("{:.4f}".format)
    print(df_cv.to_string(index=False))

    # ── 6. coeficientes Regresion Lineal ──────────────────────────
    print("\n" + "─" * 65)
    print("C) COEFICIENTES — Regresion Lineal Multiple")
    print("─" * 65)
    coef = lin_reg.named_steps["model"].coef_
    intercept = lin_reg.named_steps["model"].intercept_
    df_coef = pd.DataFrame({
        "Feature": FEATURES,
        "Coeficiente": coef,
        "Abs_Coef": np.abs(coef),
    }).sort_values("Abs_Coef", ascending=False)
    df_coef["Coeficiente"] = df_coef["Coeficiente"].map("{:+.6f}".format)
    print(f"Intercepto: {intercept:.4f}")
    print(df_coef[["Feature", "Coeficiente"]].to_string(index=False))

    print("\n" + "─" * 65)
    print("D) COEFICIENTES — Ridge")
    print("─" * 65)
    coef_r = ridge.named_steps["model"].coef_
    intercept_r = ridge.named_steps["model"].intercept_
    df_coef_r = pd.DataFrame({
        "Feature": FEATURES,
        "Coeficiente": coef_r,
        "Abs_Coef": np.abs(coef_r),
    }).sort_values("Abs_Coef", ascending=False)
    df_coef_r["Coeficiente"] = df_coef_r["Coeficiente"].map("{:+.6f}".format)
    print(f"Intercepto: {intercept_r:.4f}")
    print(df_coef_r[["Feature", "Coeficiente"]].to_string(index=False))

    # ── 7. GRAFICOS ───────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("E) GENERANDO GRAFICOS")
    print("─" * 65)

    colores = {
        "Baseline (media)": "#aaaaaa",
        "Regresion Lineal" : "#2196F3",
        "Ridge (alpha=1)"  : "#E91E63",
    }

    # ── 7a. Distribucion del indice_salud ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.histplot(y, kde=True, bins=40, color="#4CAF50", ax=axes[0])
    axes[0].set_title("Distribucion de indice_salud", fontweight="bold")
    axes[0].set_xlabel("indice_salud")

    orden = ["High Stress", "Moderate Stress", "Healthy"]
    bp_data = [df[df["Plant_Health_Status"] == cls]["indice_salud"] for cls in orden]
    bp = axes[1].boxplot(bp_data, tick_labels=orden, patch_artist=True,
                         notch=False, vert=True)
    colores_box = ["#F44336", "#FF9800", "#4CAF50"]
    for patch, color in zip(bp["boxes"], colores_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title("indice_salud por clase original", fontweight="bold")
    axes[1].set_ylabel("indice_salud")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    path = os.path.join(FIGS, "07_indice_salud_distribucion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")

    # ── 7b. Predicho vs Real (3 modelos) ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    lim = (y_test.min() - 2, y_test.max() + 2)

    for ax, (nombre, y_pred) in zip(axes, predicciones.items()):
        r2  = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        color = colores[nombre]
        ax.scatter(y_test, y_pred, alpha=0.45, s=25, color=color, edgecolors="none")
        ax.plot(lim, lim, "k--", lw=1.2, label="y = x")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Real"); ax.set_ylabel("Predicho")
        ax.set_title(f"{nombre}\nR²={r2:.4f}  RMSE={rmse:.4f}", fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)

    plt.suptitle("Predicho vs Real — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGS, "08_predicho_vs_real.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")

    # ── 7c. Analisis de residuos — Regresion Lineal ───────────────
    y_pred_lr = predicciones["Regresion Lineal"]
    residuos  = y_test.values - y_pred_lr

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # residuos vs predicho
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.scatter(y_pred_lr, residuos, alpha=0.45, s=25, color="#2196F3", edgecolors="none")
    ax1.axhline(0, color="red", lw=1.5, linestyle="--")
    ax1.set_xlabel("Valores Predichos"); ax1.set_ylabel("Residuos")
    ax1.set_title("Residuos vs Predichos", fontweight="bold")
    # banda +/-2*std
    std_res = residuos.std()
    ax1.axhline( 2 * std_res, color="orange", lw=1, linestyle=":", label="+/- 2 std")
    ax1.axhline(-2 * std_res, color="orange", lw=1, linestyle=":")
    ax1.legend(fontsize=9)

    # histograma de residuos
    ax2 = fig.add_subplot(gs[0, 2])
    sns.histplot(residuos, kde=True, bins=30, ax=ax2, color="#2196F3")
    ax2.set_title("Distribucion de Residuos", fontweight="bold")
    ax2.set_xlabel("Residuo")

    # QQ plot
    ax3 = fig.add_subplot(gs[1, :2])
    (osm, osr), (slope, intercept_qq, r) = stats.probplot(residuos, dist="norm")
    ax3.scatter(osm, osr, s=20, alpha=0.5, color="#2196F3", edgecolors="none")
    x_line = np.array([osm[0], osm[-1]])
    ax3.plot(x_line, slope * x_line + intercept_qq, "r--", lw=1.5)
    ax3.set_xlabel("Cuantiles Teoricos (Normal)"); ax3.set_ylabel("Cuantiles Observados")
    ax3.set_title(f"QQ-Plot de Residuos  (r={r:.4f})", fontweight="bold")

    # residuos vs orden
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(residuos, alpha=0.5, color="#2196F3", lw=0.8)
    ax4.axhline(0, color="red", lw=1.2, linestyle="--")
    ax4.set_xlabel("Indice de muestra"); ax4.set_ylabel("Residuo")
    ax4.set_title("Residuos en orden de test", fontweight="bold")

    plt.suptitle("Analisis de Residuos — Regresion Lineal Multiple", fontsize=13, fontweight="bold")
    path = os.path.join(FIGS, "09_analisis_residuos_lineal.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")

    # ── 7d. Coeficientes Lineal vs Ridge ──────────────────────────
    coef_lr  = lin_reg.named_steps["model"].coef_
    coef_ri  = ridge.named_steps["model"].coef_
    x_pos    = np.arange(len(FEATURES))
    width    = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x_pos - width/2, coef_lr, width, label="Lineal", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, coef_ri, width, label="Ridge",  color="#E91E63", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(FEATURES, rotation=35, ha="right", fontsize=9)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_ylabel("Coeficiente (espacio escalado)")
    ax.set_title("Coeficientes — Regresion Lineal vs Ridge", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGS, "10_coeficientes_lineal_vs_ridge.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")

    # ── 7e. CV boxplots ───────────────────────────────────────────
    cv_scores_all = {}
    for nombre, pipe in modelos.items():
        cv_res = cross_validate(
            pipe, X_train, y_train,
            cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED),
            scoring="r2"
        )
        cv_scores_all[nombre] = cv_res["test_score"]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_box = ["#aaaaaa", "#2196F3", "#E91E63"]
    bp2 = ax.boxplot(list(cv_scores_all.values()),
                     tick_labels=list(cv_scores_all.keys()),
                     patch_artist=True, notch=False)
    for patch, color in zip(bp2["boxes"], colors_box):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_ylabel("R2")
    ax.set_title(f"R2 en Validacion Cruzada ({N_FOLDS}-Fold)", fontweight="bold")
    ax.tick_params(axis="x", rotation=10)
    plt.tight_layout()
    path = os.path.join(FIGS, "11_cv_r2_boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")

    print("\nTodos los graficos guardados en reports/figures/")
    print("=" * 65)


if __name__ == "__main__":
    main()
