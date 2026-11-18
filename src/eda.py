"""
Exploratory Data Analysis - plant_health_data.csv
Saves all figures to reports/figures/
"""

import os
import sys

# Forzar UTF-8 en stdout para evitar errores cp1252 en Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # sin GUI, apto para scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── rutas ────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, "data", "raw", "plant_health_data.csv")
FIGS   = os.path.join(BASE, "reports", "figures")
os.makedirs(FIGS, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("ANÁLISIS EXPLORATORIO — Plant Health Dataset")
print("=" * 65)

df = pd.read_csv(DATA, parse_dates=["Timestamp"])
print(f"\n✔ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. COLUMNAS Y TIPOS
# ─────────────────────────────────────────────────────────────────────────────
print("── COLUMNAS Y TIPOS ─────────────────────────────────────────")
dtype_df = pd.DataFrame({
    "Columna": df.columns,
    "Dtype"  : df.dtypes.values,
}).reset_index(drop=True)
print(dtype_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 3. VALORES NULOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── VALORES NULOS ────────────────────────────────────────────")
nulos = df.isnull().sum()
pct   = (nulos / len(df) * 100).round(2)
null_df = pd.DataFrame({"Nulos": nulos, "% Nulos": pct})
null_df = null_df[null_df["Nulos"] > 0]
if null_df.empty:
    print("✔ No hay valores nulos.")
else:
    print(null_df.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 4. DUPLICADOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── DUPLICADOS ───────────────────────────────────────────────")
dups = df.duplicated().sum()
print(f"Filas duplicadas: {dups:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. ESTADÍSTICAS DESCRIPTIVAS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── ESTADÍSTICAS DESCRIPTIVAS (numéricas) ────────────────────")
num_cols = df.select_dtypes(include=np.number).columns.tolist()
desc = df[num_cols].describe().T
desc["skew"]     = df[num_cols].skew().values
desc["kurtosis"] = df[num_cols].kurtosis().values
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
print(desc.to_string())

print("\n── VARIABLE CATEGÓRICA ──────────────────────────────────────")
print(df["Plant_Health_Status"].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 6. DISTRIBUCIONES — histogramas + KDE
# ─────────────────────────────────────────────────────────────────────────────
print("\n── GENERANDO GRÁFICOS ───────────────────────────────────────")

# 6a. Histogramas de todas las variables numéricas
fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    ax = axes[i]
    sns.histplot(df[col], kde=True, ax=ax, color="steelblue", bins=40)
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.set_xlabel("")

# Ocultar ejes sobrantes
for j in range(len(num_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Distribuciones de Variables Numéricas", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
path = os.path.join(FIGS, "01_histogramas_distribucion.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✔ {path}")

# 6b. Boxplots por Plant_Health_Status
env_cols  = ["Soil_Moisture", "Ambient_Temperature", "Soil_Temperature",
             "Humidity", "Light_Intensity", "Soil_pH"]
phys_cols = ["Nitrogen_Level", "Phosphorus_Level", "Potassium_Level",
             "Chlorophyll_Content", "Electrochemical_Signal"]

for group_name, cols in [("Ambientales", env_cols), ("Fisiologicas", phys_cols)]:
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]
    palette = sns.color_palette("Set2", df["Plant_Health_Status"].nunique())
    for ax, col in zip(axes, cols):
        sns.boxplot(data=df, x="Plant_Health_Status", y=col,
                    palette=palette, ax=ax, order=sorted(df["Plant_Health_Status"].unique()))
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
    plt.suptitle(f"Boxplots — Variables {group_name} por Estado de Salud",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGS, f"02_boxplots_{group_name.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ {path}")

# 6c. Mapa de correlación
fig, ax = plt.subplots(figsize=(11, 9))
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("Mapa de Correlación — Variables Numéricas", fontsize=12, fontweight="bold")
plt.tight_layout()
path = os.path.join(FIGS, "03_correlacion_heatmap.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✔ {path}")

# 6d. Conteo de Plant_Health_Status
fig, ax = plt.subplots(figsize=(7, 4))
order = df["Plant_Health_Status"].value_counts().index
sns.countplot(data=df, x="Plant_Health_Status", order=order,
              palette="Set2", ax=ax)
ax.set_title("Distribución de Plant_Health_Status", fontsize=12, fontweight="bold")
ax.set_xlabel("")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=9)
plt.tight_layout()
path = os.path.join(FIGS, "04_target_countplot.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✔ {path}")

# 6e. Pairplot de variables fisiológicas (muestra aleatoria para velocidad)
sample = df.sample(min(2000, len(df)), random_state=42)
g = sns.pairplot(sample[phys_cols + ["Plant_Health_Status"]],
                 hue="Plant_Health_Status", diag_kind="kde",
                 plot_kws={"alpha": 0.4, "s": 15},
                 palette="Set2")
g.fig.suptitle("Pairplot — Variables Fisiológicas", y=1.02, fontsize=11, fontweight="bold")
path = os.path.join(FIGS, "05_pairplot_fisiologicas.png")
g.fig.savefig(path, dpi=130, bbox_inches="tight")
plt.close()
print(f"  ✔ {path}")

# 6f. Pairplot de variables ambientales
g2 = sns.pairplot(sample[env_cols + ["Plant_Health_Status"]],
                  hue="Plant_Health_Status", diag_kind="kde",
                  plot_kws={"alpha": 0.4, "s": 15},
                  palette="Set2")
g2.fig.suptitle("Pairplot — Variables Ambientales", y=1.02, fontsize=11, fontweight="bold")
path = os.path.join(FIGS, "06_pairplot_ambientales.png")
g2.fig.savefig(path, dpi=130, bbox_inches="tight")
plt.close()
print(f"  ✔ {path}")

print("\n✔ Todos los gráficos guardados en reports/figures/")
print("=" * 65)
