"""
tests/test_diagnostico.py  — v2
================================
Pruebas unitarias para src/diagnostico.py con indice_salud v2.
Ejecutar: pytest tests/test_diagnostico.py -v

Cambios respecto a v1:
  - Fixtures actualizadas con las 11 variables que usa config v2.
  - test_planta_sana_variables_ok_completas: espera 11 variables OK.
  - test_detalle_scores_contiene_todas_las_variables: 11 claves.
  - 6 tests nuevos para los umbrales del Arbol de Decision y
    validacion de que los rangos SM/N del DT producen los scores correctos.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diagnostico import (
    score_trapezoidal,
    cargar_config,
    categorizar,
    diagnosticar,
    ResultadoDiagnostico,
    UMBRAL_EXCELENTE,
    UMBRAL_ATENCION,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

# =============================================================================
# FIXTURES DE PLANTAS (v2 — 11 variables)
# =============================================================================

# Planta completamente sana: todas las 11 variables en zona ideal
PLANTA_SANA = {
    "Soil_Moisture"        : 35.0,   # ideal [30, 40]  score=1.0
    "Nitrogen_Level"       : 30.0,   # ideal [20, 50]  score=1.0
    "Soil_pH"              :  6.5,   # ideal [6.0,7.0] score=1.0
    "Ambient_Temperature"  : 24.0,   # ideal [20, 28]  score=1.0
    "Potassium_Level"      : 33.0,   # ideal [25, 42]  score=1.0
    "Light_Intensity"      :700.0,   # ideal [500,900] score=1.0
    "Phosphorus_Level"     : 33.0,   # ideal [25, 42]  score=1.0
    "Chlorophyll_Content"  : 42.0,   # ideal [38, 50]  score=1.0
    "Humidity"             : 57.0,   # ideal [50, 65]  score=1.0
    "Soil_Temperature"     : 20.0,   # ideal [17, 23]  score=1.0
    "Electrochemical_Signal":  1.1,  # ideal [0.8,1.5] score=1.0
}

# Todas fuera del ideal pero dentro del rango absoluto
PLANTA_DEGRADADA = {
    "Soil_Moisture"        : 15.0,   # < ideal_min=30, > abs_min=10
    "Nitrogen_Level"       : 12.0,   # < ideal_min=20, > abs_min=10
    "Soil_pH"              :  5.7,   # < ideal_min=6.0, > abs_min=5.5
    "Ambient_Temperature"  : 18.5,   # < ideal_min=20, > abs_min=18
    "Potassium_Level"      : 12.0,   # < ideal_min=25, > abs_min=10
    "Light_Intensity"      :300.0,   # < ideal_min=500, > abs_min=200
    "Phosphorus_Level"     : 12.0,   # < ideal_min=25, > abs_min=10
    "Chlorophyll_Content"  : 25.0,   # < ideal_min=38, > abs_min=20
    "Humidity"             : 42.0,   # < ideal_min=50, > abs_min=40
    "Soil_Temperature"     : 15.5,   # < ideal_min=17, > abs_min=15
    "Electrochemical_Signal":  0.4,  # < ideal_min=0.8, > abs_min=0.0
}

# Valores muy cerca del limite absoluto inferior → indice muy bajo
PLANTA_CRITICA = {
    "Soil_Moisture"        : 10.5,
    "Nitrogen_Level"       : 10.5,
    "Soil_pH"              :  5.6,
    "Ambient_Temperature"  : 18.1,
    "Potassium_Level"      : 10.5,
    "Light_Intensity"      :210.0,
    "Phosphorus_Level"     : 10.5,
    "Chlorophyll_Content"  : 20.5,
    "Humidity"             : 40.5,
    "Soil_Temperature"     : 15.1,
    "Electrochemical_Signal":  0.05,
}

# Todos fuera del rango absoluto → score=0 en todas → indice=0
PLANTA_FUERA_ABS = {
    "Soil_Moisture"        :  5.0,   # < abs_min=10
    "Nitrogen_Level"       : 55.0,   # > abs_max=51
    "Soil_pH"              :  8.0,   # > abs_max=7.5
    "Ambient_Temperature"  : 35.0,   # > abs_max=30
    "Potassium_Level"      : 55.0,   # > abs_max=50
    "Light_Intensity"      :100.0,   # < abs_min=200
    "Phosphorus_Level"     : 55.0,   # > abs_max=50
    "Chlorophyll_Content"  : 15.0,   # < abs_min=20
    "Humidity"             : 35.0,   # < abs_min=40
    "Soil_Temperature"     : 10.0,   # < abs_min=15
    "Electrochemical_Signal":  2.5,  # > abs_max=2.0
}

# Para tests de umbrales del DT: SM=15, N=12, resto en ideal
# indice = 85*min(0.25, 0.20) + 15 = 32 → Critico
PLANTA_HIGH_STRESS_TIPICA = {
    "Soil_Moisture"        : 15.0,   # score=(15-10)/(30-10)=0.25
    "Nitrogen_Level"       : 12.0,   # score=(12-10)/(20-10)=0.20
    "Soil_pH"              :  6.5,   # ideal → score=1.0
    "Ambient_Temperature"  : 24.0,   # ideal
    "Potassium_Level"      : 33.0,   # ideal
    "Light_Intensity"      :700.0,   # ideal
    "Phosphorus_Level"     : 33.0,   # ideal
    "Chlorophyll_Content"  : 42.0,   # ideal
    "Humidity"             : 57.0,   # ideal
    "Soil_Temperature"     : 20.0,   # ideal
    "Electrochemical_Signal":  1.1,  # ideal
}

# Para test Excelente: SM=35, N=30, resto en ideal → indice=100
PLANTA_HEALTHY_TIPICA = PLANTA_SANA   # identica a sana


# =============================================================================
# TESTS: score_trapezoidal  (sin cambios — funcion pura)
# =============================================================================

class TestScoreTrapezoidal:

    def test_zona_optima_devuelve_1(self):
        s = score_trapezoidal(30.0, abs_min=10, ideal_min=25, ideal_max=42, abs_max=50)
        assert s == 1.0

    def test_en_borde_ideal_min_devuelve_1(self):
        s = score_trapezoidal(25.0, abs_min=10, ideal_min=25, ideal_max=42, abs_max=50)
        assert s == 1.0

    def test_en_borde_ideal_max_devuelve_1(self):
        s = score_trapezoidal(42.0, abs_min=10, ideal_min=25, ideal_max=42, abs_max=50)
        assert s == 1.0

    def test_en_abs_min_devuelve_0(self):
        s = score_trapezoidal(10.0, abs_min=10, ideal_min=25, ideal_max=42, abs_max=50)
        assert s == 0.0

    def test_en_abs_max_devuelve_0(self):
        s = score_trapezoidal(50.0, abs_min=10, ideal_min=25, ideal_max=42, abs_max=50)
        assert s == 0.0

    def test_por_debajo_abs_min_devuelve_0(self):
        s = score_trapezoidal(5.0, abs_min=10, ideal_min=25, ideal_max=42, abs_max=50)
        assert s == 0.0

    def test_por_encima_abs_max_devuelve_0(self):
        s = score_trapezoidal(99.0, abs_min=10, ideal_min=25, ideal_max=42, abs_max=50)
        assert s == 0.0

    def test_flanco_izquierdo_lineal(self):
        s = score_trapezoidal(17.5, abs_min=10, ideal_min=25, ideal_max=42, abs_max=50)
        assert abs(s - 0.5) < 1e-9

    def test_flanco_derecho_lineal(self):
        s = score_trapezoidal(46.0, abs_min=10, ideal_min=25, ideal_max=42, abs_max=50)
        assert abs(s - 0.5) < 1e-9

    def test_score_siempre_entre_0_y_1(self):
        for val in range(0, 60, 2):
            s = score_trapezoidal(float(val), abs_min=10, ideal_min=25, ideal_max=42, abs_max=50)
            assert 0.0 <= s <= 1.0, f"score fuera de rango para val={val}: {s}"

    def test_electrochemical_zona_optima(self):
        s = score_trapezoidal(1.1, abs_min=0.0, ideal_min=0.8, ideal_max=1.5, abs_max=2.0)
        assert s == 1.0

    def test_electrochemical_flanco_izquierdo(self):
        s = score_trapezoidal(0.4, abs_min=0.0, ideal_min=0.8, ideal_max=1.5, abs_max=2.0)
        assert abs(s - 0.5) < 1e-9

    # ── Nuevos tests para umbrales del Arbol de Decision ──────────────────────

    def test_soil_moisture_umbral_dt_20_score_05(self):
        """SM=20 (umbral DT High Stress) → score=0.5 exacto en el flanco izq."""
        # Config v2: abs_min=10, ideal_min=30, ideal_max=40, abs_max=41
        s = score_trapezoidal(20.0, abs_min=10.0, ideal_min=30.0,
                               ideal_max=40.0, abs_max=41.0)
        assert abs(s - 0.5) < 1e-9, f"esperado 0.5, obtenido {s}"

    def test_soil_moisture_umbral_dt_30_score_1(self):
        """SM=30 (umbral DT Healthy) → score=1.0 (inicio del optimo)."""
        s = score_trapezoidal(30.0, abs_min=10.0, ideal_min=30.0,
                               ideal_max=40.0, abs_max=41.0)
        assert s == 1.0

    def test_nitrogen_umbral_dt_15_score_05(self):
        """N=15 (umbral DT High Stress con SM<=30) → score=0.5."""
        # Config v2: abs_min=10, ideal_min=20, ideal_max=50, abs_max=51
        s = score_trapezoidal(15.0, abs_min=10.0, ideal_min=20.0,
                               ideal_max=50.0, abs_max=51.0)
        assert abs(s - 0.5) < 1e-9, f"esperado 0.5, obtenido {s}"

    def test_nitrogen_umbral_dt_20_score_1(self):
        """N=20 (umbral DT Healthy) → score=1.0 (inicio del optimo)."""
        s = score_trapezoidal(20.0, abs_min=10.0, ideal_min=20.0,
                               ideal_max=50.0, abs_max=51.0)
        assert s == 1.0


# =============================================================================
# TESTS: categorizar
# =============================================================================

class TestCategorizar:

    def test_excelente_en_umbral(self):
        assert UMBRAL_EXCELENTE == 90.0
        assert categorizar(UMBRAL_EXCELENTE) == "Excelente"

    def test_excelente_por_encima(self):
        assert categorizar(100.0) == "Excelente"
        assert categorizar(95.0) == "Excelente"

    def test_atencion_en_umbral_inferior(self):
        assert UMBRAL_ATENCION == 50.0
        assert categorizar(UMBRAL_ATENCION) == "Atencion Requerida"

    def test_atencion_en_rango_medio(self):
        assert categorizar(60.0) == "Atencion Requerida"
        assert categorizar(89.9999) == "Atencion Requerida"

    def test_critico_en_cero(self):
        assert categorizar(0.0) == "Critico"

    def test_critico_justo_por_debajo(self):
        assert categorizar(UMBRAL_ATENCION - 0.01) == "Critico"
        assert categorizar(20.0) == "Critico"

    def test_categorias_son_string(self):
        for val in [0, 49, 50, 89, 90, 100]:
            assert isinstance(categorizar(float(val)), str)


# =============================================================================
# TESTS: diagnosticar  (v2 — 11 variables)
# =============================================================================

VARS_ESPERADAS_V2 = {
    "Soil_Moisture", "Nitrogen_Level", "Soil_pH", "Ambient_Temperature",
    "Potassium_Level", "Light_Intensity", "Phosphorus_Level",
    "Chlorophyll_Content", "Humidity", "Soil_Temperature",
    "Electrochemical_Signal",
}

class TestDiagnosticar:

    # ── Planta sana ───────────────────────────────────────────────────────────
    def test_planta_sana_indice_alto(self):
        r = diagnosticar(PLANTA_SANA, ruta_config=CONFIG_PATH)
        assert r.indice_salud == pytest.approx(100.0, abs=1e-3)

    def test_planta_sana_categoria_excelente(self):
        r = diagnosticar(PLANTA_SANA, ruta_config=CONFIG_PATH)
        assert r.categoria == "Excelente"

    def test_planta_sana_sin_alertas(self):
        r = diagnosticar(PLANTA_SANA, ruta_config=CONFIG_PATH)
        assert len(r.alertas) == 0
        assert not r.tiene_alertas

    def test_planta_sana_variables_ok_completas(self):
        """Config v2 tiene 11 variables — todas deben estar en OK."""
        r = diagnosticar(PLANTA_SANA, ruta_config=CONFIG_PATH)
        assert len(r.variables_ok) == 11

    # ── Planta degradada ──────────────────────────────────────────────────────
    def test_planta_degradada_genera_alertas(self):
        r = diagnosticar(PLANTA_DEGRADADA, ruta_config=CONFIG_PATH)
        assert r.tiene_alertas
        assert len(r.alertas) == 11   # todas fuera del ideal

    def test_planta_degradada_indice_menor_que_sana(self):
        r_sana = diagnosticar(PLANTA_SANA,      ruta_config=CONFIG_PATH)
        r_deg  = diagnosticar(PLANTA_DEGRADADA, ruta_config=CONFIG_PATH)
        assert r_deg.indice_salud < r_sana.indice_salud

    def test_planta_degradada_en_rango_absoluto(self):
        r = diagnosticar(PLANTA_DEGRADADA, ruta_config=CONFIG_PATH)
        for alerta in r.alertas:
            assert alerta.en_rango_absoluto is True, \
                f"{alerta.nombre} se marco como fuera de absoluto inesperadamente"

    def test_planta_degradada_scores_entre_0_y_1(self):
        r = diagnosticar(PLANTA_DEGRADADA, ruta_config=CONFIG_PATH)
        for col, s in r.detalle_scores.items():
            assert 0.0 <= s <= 1.0, f"score fuera de rango en {col}: {s}"

    # ── Planta critica ────────────────────────────────────────────────────────
    def test_planta_critica_categoria(self):
        r = diagnosticar(PLANTA_CRITICA, ruta_config=CONFIG_PATH)
        assert r.categoria == "Critico"

    def test_planta_critica_indice_bajo(self):
        r = diagnosticar(PLANTA_CRITICA, ruta_config=CONFIG_PATH)
        assert r.indice_salud < 20.0

    # ── Fuera de rango absoluto ───────────────────────────────────────────────
    def test_fuera_abs_score_cero(self):
        r = diagnosticar(PLANTA_FUERA_ABS, ruta_config=CONFIG_PATH)
        for col, s in r.detalle_scores.items():
            assert s == 0.0, f"esperado score=0 en {col}, obtenido {s}"

    def test_fuera_abs_indice_cero(self):
        r = diagnosticar(PLANTA_FUERA_ABS, ruta_config=CONFIG_PATH)
        assert r.indice_salud == pytest.approx(0.0, abs=1e-6)

    def test_fuera_abs_categoria_critico(self):
        r = diagnosticar(PLANTA_FUERA_ABS, ruta_config=CONFIG_PATH)
        assert r.categoria == "Critico"

    def test_fuera_abs_no_en_rango_absoluto(self):
        r = diagnosticar(PLANTA_FUERA_ABS, ruta_config=CONFIG_PATH)
        for alerta in r.alertas:
            assert alerta.en_rango_absoluto is False

    # ── Resultado es ResultadoDiagnostico ─────────────────────────────────────
    def test_retorna_tipo_correcto(self):
        r = diagnosticar(PLANTA_SANA, ruta_config=CONFIG_PATH)
        assert isinstance(r, ResultadoDiagnostico)

    def test_indice_en_rango_0_100(self):
        for planta in [PLANTA_SANA, PLANTA_DEGRADADA, PLANTA_CRITICA, PLANTA_FUERA_ABS]:
            r = diagnosticar(planta, ruta_config=CONFIG_PATH)
            assert 0.0 <= r.indice_salud <= 100.0, \
                f"indice_salud fuera de [0,100]: {r.indice_salud}"

    def test_resumen_es_string(self):
        r = diagnosticar(PLANTA_SANA, ruta_config=CONFIG_PATH)
        resumen = r.resumen()
        assert isinstance(resumen, str)
        assert "Indice de Salud" in resumen
        assert "Excelente" in resumen
        assert r.variables_limitantes == ("Soil_Moisture", "Nitrogen_Level")
        assert "Variable(s) limitante(s): Soil_Moisture, Nitrogen_Level" in resumen

    # ── Error cuando falta una variable ───────────────────────────────────────
    def test_error_variable_faltante(self):
        planta_incompleta = {
            "Chlorophyll_Content"   : 42.0,
            "Electrochemical_Signal": 1.0,
            # Faltan Soil_Moisture, Nitrogen_Level y mas
        }
        with pytest.raises(KeyError):
            diagnosticar(planta_incompleta, ruta_config=CONFIG_PATH)

    # ── Propiedades del dataclass ──────────────────────────────────────────────
    def test_es_critico_property(self):
        r = diagnosticar(PLANTA_FUERA_ABS, ruta_config=CONFIG_PATH)
        assert r.es_critico is True

    def test_no_es_critico_si_sana(self):
        r = diagnosticar(PLANTA_SANA, ruta_config=CONFIG_PATH)
        assert r.es_critico is False

    def test_detalle_scores_contiene_todas_las_variables(self):
        """Config v2 — deben aparecer las 11 variables en detalle_scores."""
        r = diagnosticar(PLANTA_SANA, ruta_config=CONFIG_PATH)
        assert set(r.detalle_scores.keys()) == VARS_ESPERADAS_V2

    # ── Alertas tienen informacion correcta ───────────────────────────────────
    def test_alertas_tienen_campos_correctos(self):
        r = diagnosticar(PLANTA_DEGRADADA, ruta_config=CONFIG_PATH)
        for alerta in r.alertas:
            assert hasattr(alerta, "nombre")
            assert hasattr(alerta, "valor")
            assert hasattr(alerta, "ideal_min")
            assert hasattr(alerta, "ideal_max")
            assert hasattr(alerta, "score")
            assert hasattr(alerta, "en_rango_absoluto")

    def test_alerta_soil_moisture_valores_correctos(self):
        """Verifica que la alerta de SM tenga los valores del config v2."""
        r = diagnosticar(PLANTA_DEGRADADA, ruta_config=CONFIG_PATH)
        sm_alerta = next(a for a in r.alertas if a.nombre == "Soil_Moisture")
        assert sm_alerta.ideal_min == pytest.approx(30.0)
        assert sm_alerta.ideal_max == pytest.approx(40.0)
        assert sm_alerta.valor     == pytest.approx(15.0)
        # score flanco izquierdo: (15-10)/(30-10) = 0.25
        assert sm_alerta.score     == pytest.approx(0.25, abs=1e-6)

    def test_alerta_nitrogen_valores_correctos(self):
        """Verifica que la alerta de NL tenga los valores del config v2."""
        r = diagnosticar(PLANTA_DEGRADADA, ruta_config=CONFIG_PATH)
        n_alerta = next(a for a in r.alertas if a.nombre == "Nitrogen_Level")
        assert n_alerta.ideal_min == pytest.approx(20.0)
        assert n_alerta.ideal_max == pytest.approx(50.0)
        assert n_alerta.valor     == pytest.approx(12.0)
        # score flanco izquierdo: (12-10)/(20-10) = 0.20
        assert n_alerta.score     == pytest.approx(0.20, abs=1e-6)

    # ── Nuevos tests de coherencia con umbrales del DT ────────────────────────

    def test_high_stress_tipico_es_critico(self):
        """
        SM=15, N=12, resto en ideal:
        indice = 85*min(0.25, 0.20) + 15 = 32 < 50 → Critico
        """
        r = diagnosticar(PLANTA_HIGH_STRESS_TIPICA, ruta_config=CONFIG_PATH)
        assert r.categoria == "Critico", \
            f"indice={r.indice_salud:.2f} no es Critico como se esperaba"
        assert r.indice_salud == pytest.approx(32.0)
        assert r.variables_limitantes == ("Nitrogen_Level",)

    def test_healthy_tipico_es_excelente(self):
        """SM=35, N=30, resto en ideal → indice=100 → Excelente."""
        r = diagnosticar(PLANTA_HEALTHY_TIPICA, ruta_config=CONFIG_PATH)
        assert r.categoria == "Excelente"
        assert r.indice_salud >= 90.0

    def test_nitrogeno_limita_con_humedad_ideal(self):
        """Nitrógeno aporta 85 × 0,05 y las otras nueve variables aportan cero."""
        planta_solo_sm_ideal = {**PLANTA_FUERA_ABS, "Soil_Moisture": 35.0, "Nitrogen_Level": 10.5}
        r = diagnosticar(planta_solo_sm_ideal, ruta_config=CONFIG_PATH)
        assert r.indice_salud == pytest.approx(4.25)
        assert r.categoria == "Critico"
        assert r.variables_limitantes == ("Nitrogen_Level",)

    def test_humedad_limita_el_indice_en_20(self):
        """Humedad aporta 85 × 0,5 y las otras nueve variables aportan 15."""
        planta_sm_20 = {**PLANTA_SANA, "Soil_Moisture": 20.0}
        r = diagnosticar(planta_sm_20, ruta_config=CONFIG_PATH)
        assert r.indice_salud == pytest.approx(57.5)
        assert r.categoria == "Atencion Requerida"
        assert r.variables_limitantes == ("Soil_Moisture",)

    def test_config_alternativa_conserva_la_suma_y_sus_umbrales(self):
        config = cargar_config(CONFIG_PATH)
        config["formula"] = "suma"
        config["umbrales_categoria"]["Excelente"] = 65
        planta = {**PLANTA_SANA, "Soil_Moisture": 20.0}
        r = diagnosticar(planta, config=config)
        assert r.indice_salud == pytest.approx(66.0)
        assert r.categoria == "Excelente"
