"""
Este script compara el precio de opciones Call asiáticas con precio promedio (media aritmética y geométrica)
frente al de una opción Call europea vanilla, variando el precio inicial del activo subyacente (S0).

Genera una gráfica con la evolución de los precios y el payoff asociado.

Script desarrollado para el TFM del Máster en Ingeniería Matemática de la UCM. 
Hecho por Lluís Boscà Ros.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy.stats import norm
from Asian_NM_Lib import (
    BSdf_asianCall_AvgPrice_arthm,
    EX_asianCall_AvgPrice_GEO,
    MC_anthitetic_asianCall_AvgPrice_GEO
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

# Función para calcular el valor de la opción Call Europea vanilla mediante la fórmula cerrada de Black-Scholes
def BS_CallPrice_EuroVanilla(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

# Parámetros globales
K = 105            # Precio de ejercicio (Strike)
r = 0.1            # Tipo de interés libre de riesgo anual
sigma = 0.2        # Volatilidad
xmax = 5           # Dominio espacial máximo en variable transformada
T = 1              # Tiempo hasta vencimiento (en años)
M_df = 5000        # Nº de pasos espaciales para diferencias finitas
N_df = 1000        # Nº de pasos temporales para diferencias finitas

# Inicialización de listas para almacenar resultados
C_df_arthm_vals = []
C_ex_GEO_vals = []
CallEuro_ex_vals = []

# Valores de precio inicial del subyacente a evaluar
S0_V = np.arange(0,190,10)

# Método de diferencias finitas (independiente de S0 y K)
x,W = BSdf_asianCall_AvgPrice_arthm(r,sigma,xmax,T,M_df,N_df)

for S0 in S0_V:
    # Método de diferencias finitas (media aritmética)
    idx = np.argmin(np.abs(x - K / S0))
    C_df = S0 * W[idx, 0]

    # Fórmula cerrada (media geométrica)
    C_ex = EX_asianCall_AvgPrice_GEO(S0, K, T, r, sigma)

    # Fórmula cerrada B-S Call Europea
    CallEuro = BS_CallPrice_EuroVanilla(S0, K, T, r, sigma)

    # Guardar resultados
    C_df_arthm_vals.append(C_df)
    C_ex_GEO_vals.append(C_ex)
    CallEuro_ex_vals.append(CallEuro)

    # Mostrar resultados por consola
    print(f"S0 = {S0}")
    print("--- Arithmetic ---")
    print(f"(DF) C = {C_df:.4f}")
    print("--- Geometric ---")
    print(f"(Ex) C = {C_ex:.4f}")
    print("--- Europea ---")
    print(f"(Ex) C = {CallEuro:.4f}")
    print("\n")

# Generar gráfica comparativa
plt.figure(figsize=(9, 4.5))
plt.plot(S0_V, C_df_arthm_vals, "d--", color="#1f77b4", label="Asiática Aritmética")
plt.plot(S0_V, C_ex_GEO_vals, "d--", color="#2ca02c", label="Asiática Geométrica")
plt.plot(S0_V, CallEuro_ex_vals, "o--", color="#d62728", label="Europea")
S0_V2 = np.linspace(0,180,1000)
plt.plot(S0_V2, np.maximum(S0_V2-K, 0), "k-", label="Payoff Europea")
plt.xlabel(r"$S_0$")
plt.ylabel(r"$C(0, 0)$")
plt.title("Comparación de opciones Asiáticas (precio promedio) y Europeas", pad=15)
plt.legend(loc="upper left", frameon=False)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
output_path = "resultados"
os.makedirs(output_path, exist_ok=True) # Crear directorio si no existe
plt.savefig(os.path.join(output_path, "compGRAPH_PriceVsEuro_byS0.pdf"))
plt.show()