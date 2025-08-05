"""
Este script evalúa y compara distintos métodos para la valoración de opciones asiáticas tipo Call
con strike promedio, variando el parámetro de volatilidad.

Se calculan los precios teóricos utilizando:
- Diferencias finitas con esquema de Crank-Nicolson (media aritmética)
- Simulación Montecarlo con variables antitéticas (media aritmética y geométrica)
- Fórmula cerrada para el caso de media geométrica

Genera una gráfica en PDF con la evolución del precio de la opción respecto a la volatilidad.

Script desarrollado para el TFM del Máster en Ingeniería Matemática de la UCM. 
Hecho por Lluís Boscà Ros.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from Asian_NM_Lib import (
    BSdf_asianCall_AvgStrike_arthm,
    MC_anthitetic_asianCall_AvgStrike_arthm,
    EX_asianCall_AvgStrike_GEO,
    MC_anthitetic_asianCall_AvgStrike_GEO
)

# Configuración de estilo para matplotlib (fuente y tamaños)
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "lines.markersize": 5.5,
})

# Parámetros del modelo
S0 = 100           # Precio inicial del activo subyacente
r = 0.1            # Tipo de interés libre de riesgo anual
Rmax = 5           # Dominio espacial máximo en variable transformada
T = 1              # Tiempo hasta vencimiento (en años)
M_df = 5000        # Nº de pasos espaciales para diferencias finitas
N_df = 1000        # Nº de pasos temporales para diferencias finitas
M_mc = 20000       # Nº de simulaciones para Montecarlo
N_mc = 5000        # Nº de pasos temporales por trayectoria en MC

# Inicialización de listas para almacenar resultados
C_df_arthm_vals = []
C_mc_arthm_vals = []
C_ex_GEO_vals = []
C_mc_GEO_vals = []

# Valores de volatilidad a evaluar
sigmaV = [0.05, 0.1, 0.2, 0.3, 0.4]

# Bucle principal: calcular precios para cada valor de sigma
for sigma in sigmaV:
    # Método de diferencias finitas (media aritmética)
    H0 = BSdf_asianCall_AvgStrike_arthm(r,sigma,Rmax,T,M_df,N_df)
    C_df = S0 * H0

    # Montecarlo (media aritmética)
    C_mc_arthm = MC_anthitetic_asianCall_AvgStrike_arthm(S0, T, r, sigma, N_mc, M_mc)

    # Fórmula exacta (media geométrica)
    C_ex = EX_asianCall_AvgStrike_GEO(S0, T, r, sigma)

    # Montecarlo (media geométrica)
    C_mc_geo = MC_anthitetic_asianCall_AvgStrike_GEO(S0, T, r, sigma, N_mc, M_mc)

    # Guardar resultados
    C_df_arthm_vals.append(C_df)
    C_mc_arthm_vals.append(C_mc_arthm)
    C_ex_GEO_vals.append(C_ex)
    C_mc_GEO_vals.append(C_mc_geo)

    # Mostrar resultados por consola
    print(f"Sigma = {sigma}")
    print("--- Arithmetic ---")
    print(f"(DF) C = {C_df:.4f}")
    print(f"(MC) C = {C_mc_arthm:.4f}")
    print("--- Geometric ---")
    print(f"(MC) C = {C_mc_geo:.4f}")
    print(f"(Ex) C = {C_ex:.4f}")
    print("\n")

# Generar gráfica comparativa
plt.figure(figsize=(9, 4.5))
plt.plot(sigmaV, C_df_arthm_vals, "d--", color="#1f77b4", label="Dif. Finitas (Arithm)")
plt.plot(sigmaV, C_mc_arthm_vals, "o--", color="#d62728", label="Montecarlo (Arithm)")
plt.plot(sigmaV, C_ex_GEO_vals, "d--", color="#2ca02c", label="Exacto (Geom)")
plt.plot(sigmaV, C_mc_GEO_vals, "o--", color="#000000", label="Montecarlo (Geom)")
plt.xlabel(r"$\sigma$")
plt.ylabel(r"$C(0, 0)$")
plt.title("Precio de opciones Asiáticas (strike promedio) en función de la volatilidad", pad=10)
plt.legend(loc="upper left", frameon=False)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
output_path = "resultados"
os.makedirs(output_path, exist_ok=True) # Crear directorio si no existe
plt.savefig(os.path.join(output_path, "compGRAPH_allStrike.pdf"))
plt.show()