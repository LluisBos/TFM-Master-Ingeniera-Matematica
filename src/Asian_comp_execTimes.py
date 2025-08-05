"""
Este script mide y compara el tiempo medio de ejecución de distintos métodos numéricos y
simulaciones para valorar opciones Call asiáticas.

Métodos evaluados:
- Diferencias finitas con esquema de Crank-Nicolson (media aritmética)
- Simulación Montecarlo con variables antitéticas (media aritmética y geométrica)

Se realizan múltiples ejecuciones (NUM_EXEC) para estimar el tiempo medio de cada método.
Los resultados se guardan en un archivo CSV.

Script desarrollado para el TFM del Máster en Ingeniería Matemática de la UCM. 
Hecho por Lluís Boscà Ros.
"""

import numpy as np
import os
import pandas as pd
import time
from Asian_NM_Lib import (
    BSdf_asianCall_AvgStrike_arthm,
    MC_anthitetic_asianCall_AvgStrike_arthm,
    MC_anthitetic_asianCall_AvgStrike_GEO,
    BSdf_asianCall_AvgPrice_arthm,
    MC_anthitetic_asianCall_AvgPrice_arthm,
    MC_anthitetic_asianCall_AvgPrice_GEO
)

NUM_EXEC = 100

# Parámetros globales
S0 = 100           # Precio inicial del activo
K = 110            # Precio de ejercicio (Strike)
r = 0.1            # Tipo de interés libre de riesgo anual
sigma = 0.3        # Volatilidad
Rmax = 5           # Dominio espacial máximo en variable transformada (strike promedio)
xmax = 5           # Dominio espacial máximo en variable transformada (precio promedio)
T = 1              # Tiempo hasta vencimiento (en años)
M_df = 5000        # Nº de pasos espaciales para diferencias finitas
N_df = 1000        # Nº de pasos temporales para diferencias finitas
M_mc = 20000       # Nº de simulaciones para Montecarlo
N_mc = 5000        # Nº de pasos temporales por trayectoria en MC

# Listas para guardar los tiempos de cada método
tiemposDF_strike = []
tiemposMC_arthm_strike = []
tiemposMC_geo_strike = []
tiemposDF_price = []
tiemposMC_arthm_price = []
tiemposMC_geo_price = []

for i in range(NUM_EXEC):
    # ---------- AVG STRIKE ----------
    t0 = time.time()
    H0 = BSdf_asianCall_AvgStrike_arthm(r, sigma, Rmax, T, M_df, N_df)
    C_df_strike = S0 * H0
    t1 = time.time()

    C_mc_arthm_strike = MC_anthitetic_asianCall_AvgStrike_arthm(S0, T, r, sigma, N_mc, M_mc)
    t2 = time.time()

    C_mc_geo_strike = MC_anthitetic_asianCall_AvgStrike_GEO(S0, T, r, sigma, N_mc, M_mc)
    t3 = time.time()
        
    # ---------- AVG PRICE ----------
    x, W = BSdf_asianCall_AvgPrice_arthm(r, sigma, xmax, T, M_df, N_df)
    idx = np.argmin(np.abs(x - K / S0))
    C_df_price = S0 * W[idx, 0]
    t4 = time.time()

    C_mc_arthm_price = MC_anthitetic_asianCall_AvgPrice_arthm(S0, K, T, r, sigma, N_mc, M_mc)
    t5 = time.time()

    C_mc_geo_price = MC_anthitetic_asianCall_AvgPrice_GEO(S0, K, T, r, sigma, N_mc, M_mc)
    t6 = time.time()
    
    # Guardar tiempos
    tiemposDF_strike.append(t1-t0)
    tiemposMC_arthm_strike.append(t2-t1)
    tiemposMC_geo_strike.append(t3-t2)
    tiemposDF_price.append(t4-t3)
    tiemposMC_arthm_price.append(t5-t4)
    tiemposMC_geo_price.append(t6-t5)

    print(f"Ejecución {i+1}")

# Resultados individuales
resultados = {
    'DF_Strike': tiemposDF_strike,
    'MC_Artm_Strike': tiemposMC_arthm_strike,
    'MC_Geom_Strike': tiemposMC_geo_strike,
    'DF_Price': tiemposDF_price,
    'MC_Artm_Price': tiemposMC_arthm_price,
    'MC_Geom_Price': tiemposMC_geo_price,
}

# Promedios
resultadosMEAN = {
    'DF_Strike': np.mean(tiemposDF_strike),
    'MC_Artm_Strike': np.mean(tiemposMC_arthm_strike),
    'MC_Geom_Strike': np.mean(tiemposMC_geo_strike),
    'DF_Price': np.mean(tiemposDF_price),
    'MC_Artm_Price': np.mean(tiemposMC_arthm_price),
    'MC_Geom_Price': np.mean(tiemposMC_geo_price),
}

print("Tiempos medios (en segundos):")
print(resultadosMEAN)

# Crear carpeta si no existe
output_path = "resultados"
os.makedirs(output_path, exist_ok=True)

# Guardar resultados a CSV
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(os.path.join(output_path, "Asian_tiemposExec.csv"), index=False)