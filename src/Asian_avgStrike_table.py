"""
Este script calcula y guarda en CSV los precios de opciones asiáticas tipo call
con strike promedio, evaluando múltiples combinaciones de tasa de interés y volatilidad.

Se utilizan los siguientes métodos para la valoración:
- Diferencias finitas con esquema de Crank-Nicolson (media aritmética)
- Simulaciones de Montecarlo con variables antitéticas (media aritmética y geométrica)
- Fórmula exacta cerrada para el caso de media geométrica

Script desarrollado para el TFM del Máster en Ingeniería Matemática de la UCM. 
Hecho por Lluís Boscà Ros.
"""
import numpy as np
import os
import pandas as pd
from Asian_NM_Lib import (
    BSdf_asianCall_AvgStrike_arthm,
    MC_anthitetic_asianCall_AvgStrike_arthm,
    EX_asianCall_AvgStrike_GEO,
    MC_anthitetic_asianCall_AvgStrike_GEO
)

# Parámetros globales
S0 = 100           # Precio inicial del activo
Rmax = 5           # Dominio espacial máximo en variable transformada
T = 1              # Tiempo hasta vencimiento (en años)
M_df = 5000        # Nº de pasos espaciales para diferencias finitas
N_df = 1000        # Nº de pasos temporales para diferencias finitas
M_mc = 20000       # Nº de simulaciones para Montecarlo
N_mc = 5000        # Nº de pasos temporales por trayectoria en MC

# Rangos de parámetros a evaluar
rV = [0.05, 0.1, 0.2, 0.4]                  # Tipos de interés
sigmaV = [0.05, 0.1, 0.2, 0.3, 0.4]         # Volatilidades

resultados = []

# Evaluación para cada combinación (r, sigma)
for r in rV:
    for sigma in sigmaV:
        # Diferencias finitas (media aritmética)
        H0 = BSdf_asianCall_AvgStrike_arthm(r, sigma, Rmax, T, M_df, N_df)
        C_df = S0 * H0

        # Montecarlo (media aritmética)
        C_mc_arthm = MC_anthitetic_asianCall_AvgStrike_arthm(S0, T, r, sigma, N_mc, M_mc)

        # Fórmula cerrada exacta (media geométrica)
        C_ex = EX_asianCall_AvgStrike_GEO(S0, T, r, sigma)

        # Montecarlo (media geométrica)
        C_mc_geo = MC_anthitetic_asianCall_AvgStrike_GEO(S0, T, r, sigma, N_mc, M_mc)

        # Almacenar resultados con 4 decimales
        resultados.append({
            'r': r,
            'sigma': sigma,
            'C_df': f"{C_df:.4f}",
            'C_mc_arthm': f"{C_mc_arthm:.4f}",
            'C_mc_geo': f"{C_mc_geo:.4f}",
            'C_ex': f"{C_ex:.4f}"
        })
        print(f"Caso con r= {r} y sigma = {sigma} hecho")

# Crear carpeta si no existe
output_path = "resultados"
os.makedirs(output_path, exist_ok=True)

# Guardar resultados a CSV
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(os.path.join(output_path, "AvgStrike_results.csv"), index=False)