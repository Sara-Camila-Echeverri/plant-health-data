"""
diagnostico.py
==============
Recibe los valores fisiologicos y ambientales de una planta,
calcula el indice_salud, lo categoriza y lista las variables
fuera de su rango ideal segun config.yaml.

La fórmula y los umbrales se definen en config.yaml.
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Rutas ─────────────────────────────────────────────────────────────────────
_BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG = os.path.join(_BASE, "config.yaml")

# ── Umbrales de categoria (modificables) ──────────────────────────────────────
UMBRAL_EXCELENTE = 90.0
UMBRAL_ATENCION = 50.0
VARIABLES_PRINCIPALES = ("Soil_Moisture", "Nitrogen_Level")

# ── Dataclasses de resultado ──────────────────────────────────────────────────

@dataclass
class VariableAlerta:
    """Una variable fuera de su rango ideal."""
    nombre: str
    valor: float
    ideal_min: float
    ideal_max: float
    abs_min: float
    abs_max: float
    score: float          # 0..1 calculado para esta variable
    en_rango_absoluto: bool   # False si esta fuera de viabilidad total


@dataclass
class ResultadoDiagnostico:
    """Resultado completo del diagnostico de una planta."""
    indice_salud: float
    categoria: str
    alertas: List[VariableAlerta] = field(default_factory=list)
    variables_ok: List[str]       = field(default_factory=list)
    detalle_scores: Dict[str, float] = field(default_factory=dict)
    variables_limitantes: tuple[str, ...] = ()

    # ── helpers de presentacion ───────────────────────────────────────────────
    def resumen(self) -> str:
        lineas = [
            f"Indice de Salud : {self.indice_salud:.2f} / 100",
            f"Categoria       : {self.categoria}",
            f"Variables OK    : {', '.join(self.variables_ok) if self.variables_ok else 'ninguna'}",
        ]
        if self.variables_limitantes:
            lineas.append(f"Variable(s) limitante(s): {', '.join(self.variables_limitantes)}")
        if self.alertas:
            lineas.append("Variables fuera de rango ideal:")
            for a in self.alertas:
                estado = "FUERA DE VIABILIDAD" if not a.en_rango_absoluto else "suboptimo"
                lineas.append(
                    f"  - {a.nombre}: {a.valor:.4f}  "
                    f"(ideal [{a.ideal_min}, {a.ideal_max}])  "
                    f"score={a.score:.3f}  [{estado}]"
                )
        else:
            lineas.append("Todas las variables en rango ideal.")
        return "\n".join(lineas)

    @property
    def tiene_alertas(self) -> bool:
        return len(self.alertas) > 0

    @property
    def es_critico(self) -> bool:
        return self.categoria == "Critico"


# ── Funciones principales ──────────────────────────────────────────────────────

def cargar_config(ruta_config: Optional[str] = None) -> dict:
    """Carga config.yaml y devuelve la seccion 'indice_salud'."""
    ruta = ruta_config or _CONFIG
    with open(ruta, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["indice_salud"]


def umbrales_categoria(config: dict) -> tuple[float, float]:
    """Devuelve el inicio de Atención y el inicio de Excelente, en ese orden."""
    umbrales = config.get("umbrales_categoria", {})
    return (
        float(umbrales.get("Atencion_Requerida", UMBRAL_ATENCION)),
        float(umbrales.get("Excelente", UMBRAL_EXCELENTE)),
    )


def score_trapezoidal(valor: float,
                      abs_min: float, ideal_min: float,
                      ideal_max: float, abs_max: float) -> float:
    """
    Funcion de scoring trapezoidal en [0, 1].

    Zonas:
        [abs_min,  ideal_min) -> rampa ascendente   0 → 1
        [ideal_min, ideal_max] -> optimo             1.0
        (ideal_max, abs_max]  -> rampa descendente  1 → 0
        fuera de [abs_min, abs_max] -> 0.0
    """
    if valor <= abs_min or valor >= abs_max:
        return 0.0
    if ideal_min <= valor <= ideal_max:
        return 1.0
    if valor < ideal_min:
        return (valor - abs_min) / (ideal_min - abs_min)
    return (abs_max - valor) / (abs_max - ideal_max)


def categorizar(
    indice: float, umbral_atencion: float = UMBRAL_ATENCION,
    umbral_excelente: float = UMBRAL_EXCELENTE,
) -> str:
    """Convierte el indice_salud numerico en categoria textual."""
    if indice >= umbral_excelente:
        return "Excelente"
    if indice >= umbral_atencion:
        return "Atencion Requerida"
    return "Critico"


def diagnosticar(
    valores_planta: Dict[str, float], ruta_config: Optional[str] = None,
    *, config: Optional[dict] = None,
) -> ResultadoDiagnostico:
    """
    Diagnostica el estado de una planta.

    Parameters
    ----------
    valores_planta : dict
        Mapa {nombre_columna: valor_numerico} con al menos las variables
        de sensor definidas en config.yaml.

    ruta_config : str, opcional
        Ruta alternativa a config.yaml. Si None usa la ruta por defecto.
    config : dict, opcional
        Sección indice_salud ya cargada. Si se suministra, se usa directamente.

    Returns
    -------
    ResultadoDiagnostico
    """
    cfg = cargar_config(ruta_config) if config is None else config
    variables_cfg = cfg["variables"]

    alertas: List[VariableAlerta] = []
    ok: List[str] = []
    puntajes: Dict[str, float] = {}

    for col, params in variables_cfg.items():
        if col not in valores_planta:
            raise KeyError(
                f"Variable '{col}' requerida para el diagnostico pero no "
                f"encontrada en valores_planta. Claves recibidas: "
                f"{list(valores_planta.keys())}"
            )
        val = float(valores_planta[col])
        s   = score_trapezoidal(
            valor     = val,
            abs_min   = params["abs_min"],
            ideal_min = params["ideal_min"],
            ideal_max = params["ideal_max"],
            abs_max   = params["abs_max"],
        )
        puntajes[col] = s

        if params["ideal_min"] <= val <= params["ideal_max"]:
            ok.append(col)
        else:
            en_abs = params["abs_min"] < val < params["abs_max"]
            alertas.append(VariableAlerta(
                nombre            = col,
                valor             = val,
                ideal_min         = params["ideal_min"],
                ideal_max         = params["ideal_max"],
                abs_min           = params["abs_min"],
                abs_max           = params["abs_max"],
                score             = round(s, 6),
                en_rango_absoluto = en_abs,
            ))

    principales = [nombre for nombre in VARIABLES_PRINCIPALES if nombre in puntajes]
    minimo = min((puntajes[nombre] for nombre in principales), default=None)
    limitantes = tuple(nombre for nombre in principales if puntajes[nombre] == minimo)
    formula = cfg.get("formula", "suma")
    if formula == "minimo":
        if len(principales) != len(VARIABLES_PRINCIPALES):
            raise ValueError("La fórmula minimo requiere Soil_Moisture y Nitrogen_Level.")
        aporte_principal = 85 * minimo
        aportes_restantes = sum(
            variables_cfg[nombre]["peso"] * puntaje
            for nombre, puntaje in puntajes.items()
            if nombre not in VARIABLES_PRINCIPALES
        )
        indice = round(aporte_principal + aportes_restantes, 4)
    elif formula in ("suma", "suma_ponderada_normalizada"):
        peso_total = sum(params["peso"] for params in variables_cfg.values())
        suma_ponderada = sum(
            variables_cfg[nombre]["peso"] * puntaje for nombre, puntaje in puntajes.items()
        )
        indice = round(100.0 * suma_ponderada / peso_total, 4)
    else:
        raise ValueError(f"Fórmula desconocida: {formula!r}. Usa suma o minimo.")

    return ResultadoDiagnostico(
        indice_salud    = indice,
        categoria       = categorizar(indice, *umbrales_categoria(cfg)),
        alertas         = alertas,
        variables_ok    = ok,
        detalle_scores  = {nombre: round(puntaje, 6) for nombre, puntaje in puntajes.items()},
        variables_limitantes = limitantes,
    )


# ── CLI rapido ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Planta de ejemplo
    ejemplo = {
        "Soil_Moisture": 35.0,
        "Nitrogen_Level": 18.0,
        "Soil_pH": 6.5,
        "Ambient_Temperature": 24.0,
        "Potassium_Level": 12.0,
        "Light_Intensity": 700.0,
        "Phosphorus_Level": 30.0,
        "Chlorophyll_Content": 42.0,
        "Humidity": 57.0,
        "Soil_Temperature": 20.0,
        "Electrochemical_Signal": 0.5,
    }
    resultado = diagnosticar(ejemplo)
    print(resultado.resumen())
