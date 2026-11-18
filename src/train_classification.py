"""
train_classification.py
=======================
Entrena Arbol de Decision y Random Forest para predecir Plant_Health_Status.
Features: todas las columnas excepto Timestamp y Plant_ID.
Reporta accuracy (CV), precision, recall, F1, confusion matrix
e importancia de variables.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, train_test_split
)
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "raw", "plant_health_data.csv")
FIGS = os.path.join(BASE, "reports", "figures")
os.makedirs(FIGS, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

SEED     = 42
N_FOLDS  = 5
TARGET   = "Plant_Health_Status"
EXCLUIR  = ["Timestamp", "Plant_ID", TARGET]
CLASS_ORDER = ["Healthy", "Moderate Stress", "High Stress"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def cv_clasificacion(modelo, X, y, nombre):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    res = cross_validate(
        modelo, X, y, cv=skf,
        scoring=["accuracy", "f1_macro", "f1_weighted"],
        return_train_score=True,
    )
    return {
        "Modelo"              : nombre,
        "Train Acc (media)"   : res["train_accuracy"].mean(),
        "CV Acc (media)"      : res["test_accuracy"].mean(),
        "CV Acc (std)"        : res["test_accuracy"].std(),
        "CV F1-macro (media)" : res["test_f1_macro"].mean(),
        "CV F1-macro (std)"   : res["test_f1_macro"].std(),
        "CV F1-weighted"      : res["test_f1_weighted"].mean(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("CLASIFICACION — Plant_Health_Status")
    print("=" * 70)

    # ── 1. Datos ──────────────────────────────────────────────────
    df = pd.read_csv(DATA, parse_dates=["Timestamp"])
    features = [c for c in df.columns if c not in EXCLUIR]
    X = df[features]
    y = df[TARGET]

    print(f"\nDataset: {df.shape[0]} filas")
    print(f"Features ({len(features)}): {features}")
    print(f"Target  : {TARGET}")
    print(f"\nDistribucion del target:")
    print(y.value_counts().to_string())

    # ── 2. Split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    # ── 3. Modelos ────────────────────────────────────────────────
    dt  = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,
                                  class_weight="balanced", random_state=SEED)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=None,
                                  min_samples_leaf=5, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)

    modelos = {"Arbol de Decision": dt, "Random Forest": rf}

    # ── 4. Validacion cruzada ─────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"A) VALIDACION CRUZADA ESTRATIFICADA ({N_FOLDS}-FOLD)")
    print("─" * 70)

    resultados_cv = []
    for nombre, modelo in modelos.items():
        r = cv_clasificacion(modelo, X_train, y_train, nombre)
        resultados_cv.append(r)

    df_cv = pd.DataFrame(resultados_cv)
    for col in df_cv.columns[1:]:
        df_cv[col] = df_cv[col].map("{:.4f}".format)
    print(df_cv.to_string(index=False))

    # ── 5. Metricas en test ───────────────────────────────────────
    print("\n" + "─" * 70)
    print("B) METRICAS EN TEST SET")
    print("─" * 70)

    predicciones = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        predicciones[nombre] = y_pred
        acc = accuracy_score(y_test, y_pred)
        print(f"\n--- {nombre} ---  Accuracy test: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=CLASS_ORDER))

    # ── 6. Importancia de variables ───────────────────────────────
    print("\n" + "─" * 70)
    print("C) IMPORTANCIA DE VARIABLES")
    print("─" * 70)

    for nombre, modelo in modelos.items():
        imp = pd.DataFrame({
            "Feature"   : features,
            "Importancia": modelo.feature_importances_,
        }).sort_values("Importancia", ascending=False)
        print(f"\n{nombre}:")
        imp["Importancia"] = imp["Importancia"].map("{:.6f}".format)
        print(imp.to_string(index=False))

    # ── 7. Top-3 variables determinantes ─────────────────────────
    print("\n" + "─" * 70)
    print("D) VARIABLES QUE DETERMINAN LA ETIQUETA (RF top-3)")
    print("─" * 70)
    rf_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    top3   = rf_imp.head(3)
    acum   = 0.0
    for feat, imp_val in rf_imp.items():
        acum += imp_val
        print(f"  {feat:<25} {imp_val:.4f}   (acumulado: {acum:.4f})")

    print(f"\nTop-3 variables explican el {top3.sum()*100:.1f}% de la importancia total.")
    print(f"Top-3: {list(top3.index)}")

    # ── 8. Regla del arbol ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("E) REGLAS DEL ARBOL DE DECISION (primeras 3 profundidades)")
    print("─" * 70)
    dt_shallow = DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                         random_state=SEED)
    dt_shallow.fit(X_train, y_train)
    print(export_text(dt_shallow, feature_names=features, max_depth=3))

    # ── 9. GRAFICOS ───────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("F) GENERANDO GRAFICOS")
    print("─" * 70)

    # 9a. Importancia de variables comparada
    imp_dt = pd.Series(dt.feature_importances_, index=features)
    imp_rf = pd.Series(rf.feature_importances_, index=features)
    imp_df = pd.DataFrame({"Arbol Decision": imp_dt, "Random Forest": imp_rf})
    imp_df = imp_df.sort_values("Random Forest", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos   = np.arange(len(imp_df))
    ax.barh(y_pos - 0.2, imp_df["Arbol Decision"], height=0.38,
            label="Arbol Decision", color="#FF9800", alpha=0.85)
    ax.barh(y_pos + 0.2, imp_df["Random Forest"],  height=0.38,
            label="Random Forest",  color="#4CAF50", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_df.index, fontsize=9)
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title("Importancia de Variables — DT vs RF", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    p = os.path.join(FIGS, "12_importancia_variables.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> {p}")

    # 9b. Matrices de confusion
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (nombre, y_pred) in zip(axes, predicciones.items()):
        cm  = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_ORDER)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{nombre}", fontweight="bold", fontsize=10)
        ax.tick_params(axis="x", rotation=20)
    plt.suptitle("Matrices de Confusion — Test Set", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(FIGS, "13_confusion_matrices.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> {p}")

    # 9c. CV accuracy por fold (boxplot)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores_dt = cross_validate(dt, X_train, y_train, cv=skf, scoring="accuracy")["test_score"]
    scores_rf = cross_validate(rf, X_train, y_train, cv=skf, scoring="accuracy")["test_score"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot([scores_dt, scores_rf],
                    tick_labels=["Arbol Decision", "Random Forest"],
                    patch_artist=True, notch=False)
    colors_bp = ["#FF9800", "#4CAF50"]
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy en CV ({N_FOLDS}-Fold Estratificado)", fontweight="bold")
    plt.tight_layout()
    p = os.path.join(FIGS, "14_cv_accuracy_boxplot.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> {p}")

    # 9d. Top-10 importancia RF (bar chart horizontal detallado)
    top10 = imp_rf.sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top10.index[::-1], top10.values[::-1],
                   color="#4CAF50", alpha=0.85)
    ax.bar_label(bars, fmt="{:.4f}", padding=3, fontsize=9)
    ax.set_xlabel("Importancia Gini")
    ax.set_title("Top-10 Variables — Random Forest", fontweight="bold")
    plt.tight_layout()
    p = os.path.join(FIGS, "15_rf_top10_importancia.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> {p}")

    print("\nTodos los graficos guardados en reports/figures/")
    print("=" * 70)

    return rf_imp   # devuelve importancias para uso externo


if __name__ == "__main__":
    main()
