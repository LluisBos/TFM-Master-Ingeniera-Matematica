"""
Este script calcula y almacena en CSV los precios de opciones asiáticas tipo Call
con precio promedio, considerando múltiples combinaciones de tasa de interés, volatilidad
y precios de ejercicio.

Se utilizan los siguientes métodos para la valoración:
- Diferencias finitas con esquema de Crank-Nicolson (media aritmética)
- Simulaciones de Montecarlo con variables antitéticas (media aritmética y geométrica)
- Fórmula exacta cerrada para el caso de media geométrica (Kemna y Vorst)

Script desarrollado para el TFM del Máster en Ingeniería Matemática de la UCM. 
Hecho por Lluís Boscà Ros.
"""

import numpy as np
import os
import pandas as pd
from Asian_NM_Lib import (
    BSdf_asianCall_AvgPrice_arthm,
    MC_anthitetic_asianCall_AvgPrice_arthm,
    EX_asianCall_AvgPrice_GEO,
    MC_anthitetic_asianCall_AvgPrice_GEO
)

# Parámetros globales
S0 = 100           # Precio inicial del activo subyacente
xmax = 5           # Dominio espacial máximo en variable transformada
T = 1              # Tiempo hasta vencimiento (en años)
M_df = 5000        # Nº de pasos espaciales para diferencias finitas
N_df = 1000        # Nº de pasos temporales para diferencias finitas
M_mc = 20000       # Nº de simulaciones para Montecarlo
N_mc = 5000        # Nº de pasos temporales por trayectoria en MC

# Valores a evaluar
K_V = [90, 95, 105, 110]                  # Precios de ejercicio
rV = [0.05, 0.1, 0.2, 0.4]                # Tipos de interés
sigmaV = [0.05, 0.1, 0.2, 0.3, 0.4]       # Volatilidades

resultados = []

# Cálculo de precios para cada combinación (r, sigma, K)
for r in rV:
    for sigma in sigmaV:
        x,W = BSdf_asianCall_AvgPrice_arthm(r,sigma,xmax,T,M_df,N_df)
        for K in K_V:
            # Diferencias finitas (media aritmética)
            idx = np.argmin(np.abs(x - K / S0))
            C_df = S0 * W[idx, 0]

            # Montecarlo (media aritmética)
            C_mc_arthm = MC_anthitetic_asianCall_AvgPrice_arthm(S0, K, T, r, sigma, N_mc, M_mc)

            # Fórmula cerrada exacta (media geométrica)
            C_ex = EX_asianCall_AvgPrice_GEO(S0, K, T, r, sigma)

            # Montecarlo (media geométrica)
            C_mc_geo = MC_anthitetic_asianCall_AvgPrice_GEO(S0, K, T, r, sigma, N_mc, M_mc)

            # Guardar resultados como strings con 4 decimales
            resultados.append({
                'r': r,
                'sigma': sigma,
                'K': K,
                'C_df': f"{C_df:.4f}",
                'C_mc_arthm': f"{C_mc_arthm:.4f}",
                'C_mc_geo': f"{C_mc_geo:.4f}",
                'C_ex': f"{C_ex:.4f}"
            })
            print(f"Caso con r = {r}, sigma = {sigma} y  K = {K} hecho")

# Crear carpeta si no existe
output_path = "resultados"
os.makedirs(output_path, exist_ok=True)

# Guardar resultados en CSV
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(os.path.join(output_path, "AvgPrice_results.csv"), index=False)