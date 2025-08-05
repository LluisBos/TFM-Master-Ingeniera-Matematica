"""
Módulo Asian_NM_lib

Este módulo contiene la implementación de distintos métodos numéricos y de simulación
para la valoración de opciones asiáticas tipo Call, tanto con strike promedio como 
precio promedio, usando medias aritméticas y geométricas.

Métodos implementados:
- Diferencias finitas con esquema de Crank-Nicolson para opciones con media aritmética.
- Simulación Monte Carlo con trayectorias antitéticas para medias aritméticas y geométricas.
- Fórmulas analíticas cerradas para el caso geométrico basadas en la ecuación de Black-Scholes.

Módulo desarrollado para el TFM del Máster en Ingeniería Matemática de la UCM. 
Desarrollado por Lluis Bosca Ros.
"""

import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_banded

# ******************************************************************************************** #
# ************************** OPCIONES ASIÁTICAS CON STRIKE PROMEDIO ************************** #
# ******************************************************************************************** #

# **************************************************************** #
# Método de diferencias finitas con esquema de Crank-Nicolson      #
# para opciones asiáticas con strike promedio y media aritmética   #
# **************************************************************** #
def BSdf_asianCall_AvgStrike_arthm(r,sigma,Rmax,T,M,N):
    h = Rmax / M
    tau = T / N

    t = np.linspace(0, T, N + 1)
    R = np.linspace(0, Rmax, M + 1)

    H = np.zeros((M + 1, N + 1))

    # Condiciones iniciales
    H[:, -1] = np.maximum(1 - R / T, 0)
    H[-1, :] = 0

    # Parámetros a, b, c
    i = np.arange(1, M)
    a = tau / 4 * ((1 / h - i * r) - (i ** 2) * sigma ** 2)
    b = tau / 2 * (i ** 2) * sigma ** 2
    c = -tau / 4 * ((1 / h - i * r) + (i ** 2) * sigma ** 2)

    # Construcción de la matriz tridiagonal I+B en forma compacta para solve_banded
    I_B = np.zeros((3, M - 1))
    I_B[0, 1:] = c[:-1]      # superdiagonal
    I_B[1, :] = 1 + b        # diagonal principal
    I_B[2, :-1] = a[1:]      # subdiagonal

    # Esquema C-N regresivo
    for j in range(N, 0, -1):
        # Frontera izquierda: fórmula explícita
        H[0, j - 1] = (1 - tau / h) * H[0, j] + tau / h * H[1, j]
        dC = np.zeros(M - 1)
        dC[0] = -a[0] * (H[0, j] + H[0, j - 1])
        RHS = 2 * H[1:M, j] + dC
        H[1:M, j - 1] = solve_banded((1, 1), I_B, RHS) - H[1:M, j]

    H0 = H[0, 0]

    return H0


# **************************************************************** #
# Método de simulaciones de Montecarlo con variables antitéticas   #
# para opciones asiáticas con strike promedio y media aritmética   #
# **************************************************************** #
def MC_anthitetic_asianCall_AvgStrike_arthm(S0, T, r, sigma, N, M):
    tau = T / N
    M2 = M // 2

    # Generación de M/2 trayectorias normales y sus trayectorias antitéticas
    Z = np.random.normal(size=(M2, N))
    Z_antithetic = -Z

    # Cálculo de las trayectorias
    drift = (r - 0.5 * sigma ** 2) * tau
    diffusion = sigma * np.sqrt(tau)
    increments = drift + diffusion * Z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack((np.zeros((M2, 1)), log_paths))
    S = S0 * np.exp(log_paths)

    # Cálculo del promedio aritmético y payoff descontado
    I = np.mean(S[:, 1:], axis=1)
    payoff = np.maximum(S[:, -1] - I, 0)
    discounted1 = np.exp(-r * T) * payoff

    # Cálculo de las trayectorias antitéticas
    increments_a = drift + diffusion * Z_antithetic
    log_paths_a = np.cumsum(increments_a, axis=1)
    log_paths_a = np.hstack((np.zeros((M2, 1)), log_paths_a))
    S_a = S0 * np.exp(log_paths_a)

    # Cálculo del promedio aritmético y payoff descontado para antitéticas
    I_a = np.mean(S_a[:, 1:], axis=1)
    payoff_a = np.maximum(S_a[:, -1] - I_a, 0)
    discounted2 = np.exp(-r * T) * payoff_a

    # Promedio de ambas simulaciones
    total = 0.5 * (discounted1 + discounted2)

    return np.mean(total)


# **************************************************************** #
# Método de simulaciones de Montecarlo con variables antitéticas   #
# para opciones asiáticas con strike promedio y media geométrica   #
# **************************************************************** #
def MC_anthitetic_asianCall_AvgStrike_GEO(S0, T, r, sigma, N, M):
    tau = T / N
    M2 = M // 2 

    # Generación de M/2 trayectorias normales y sus trayectorias antitéticas
    Z = np.random.normal(size=(M2, N))
    Z_antithetic = -Z

    # Cálculo de las trayectorias
    drift = (r - 0.5 * sigma ** 2) * tau
    diffusion = sigma * np.sqrt(tau)
    increments = drift + diffusion * Z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack((np.zeros((M2, 1)), log_paths)) 
    S = S0 * np.exp(log_paths)

    # Cálculo del promedio geométrico y payoff descontado
    log_avg = np.mean(np.log(S[:, 1:]), axis=1)
    G = np.exp(log_avg)
    payoff = np.maximum(S[:, -1] - G, 0)
    discounted1 = np.exp(-r * T) * payoff

    # Cálculo de las trayectorias antitéticas
    increments_a = drift + diffusion * Z_antithetic
    log_paths_a = np.cumsum(increments_a, axis=1)
    log_paths_a = np.hstack((np.zeros((M2, 1)), log_paths_a))
    S_a = S0 * np.exp(log_paths_a)

    # Cálculo del promedio geométrico y payoff descontado para antitéticas
    log_avg_a = np.mean(np.log(S_a[:, 1:]), axis=1)
    G_a = np.exp(log_avg_a)
    payoff_a = np.maximum(S_a[:, -1] - G_a, 0)
    discounted2 = np.exp(-r * T) * payoff_a

    # Promedio de ambas simulaciones
    total = 0.5 * (discounted1 + discounted2)

    return np.mean(total)

# **************************************************************** #
# Fórmula cerrada de la ecuación de Black-Scholes adaptada         #
# para opciones asiáticas con strike promedio y media geométrica   #
# **************************************************************** #
def EX_asianCall_AvgStrike_GEO(S0, T, r, sigma):
    d1 = (-0.5*(r - 1/6 * sigma**2)*T) / (sigma * np.sqrt(T/3))
    d2 = d1 - sigma*np.sqrt(T/3)
    return S0 * norm.cdf(-d2) - S0 * np.exp(-T/2 * (r + (sigma**2)/6)) * norm.cdf(-d1)



# ******************************************************************************************** #
# ************************** OPCIONES ASIÁTICAS CON PRECIO PROMEDIO ************************** #
# ******************************************************************************************** #

# **************************************************************** #
# Método de diferencias finitas con esquema de Crank-Nicolson      #
# para opciones asiáticas con precio promedio y media aritmética   #
# **************************************************************** #
def BSdf_asianCall_AvgPrice_arthm(r,sigma,xmax,T,M,N):
    h = xmax / M
    tau = T / N

    t = np.linspace(0, T, N + 1)
    x = np.linspace(0, xmax, M + 1)

    W = np.zeros((M + 1, N + 1))

    # Condiciones iniciales
    W[:, -1] = np.maximum(-x, 0)
    W[0, :] = 1/(r*T) * (1 - np.exp(-r*(T-t)))
    W[-1, :] = 0

    # Parámetros a, b, c
    i = np.arange(1, M)
    a = -tau / 4 * ((1 / (T*h) + i * r) + (i ** 2) * sigma ** 2)
    b = tau / 2 * (i ** 2) * sigma ** 2
    c = tau / 4 * ((1 / (T*h) + i * r) - (i ** 2) * sigma ** 2)
    
    # Construcción de la matriz tridiagonal I+B en forma compacta para solve_banded
    I_B = np.zeros((3, M - 1))
    I_B[0, 1:] = c[:-1]      # superdiagonal
    I_B[1, :] = 1 + b        # diagonal principal
    I_B[2, :-1] = a[1:]      # subdiagonal

    # Esquema C-N regresivo
    for j in range(N, 0, -1):
        dC = np.zeros(M - 1)
        dC[0] = -a[0] * (W[0, j] + W[0, j - 1])
        RHS = 2 * W[1:M, j] + dC
        W[1:M, j - 1] = solve_banded((1, 1), I_B, RHS) - W[1:M, j]
    
    return x,W


# **************************************************************** #
# Método de simulaciones de Montecarlo con variables antitéticas   #
# para opciones asiáticas con precio promedio y media aritmética   #
# **************************************************************** #
def MC_anthitetic_asianCall_AvgPrice_arthm(S0, K, T, r, sigma, N, M):
    tau = T / N
    M2 = M // 2

    # Generación M/2 trayectorias normales y sus trayectorias antitéticas
    Z = np.random.normal(size=(M2, N))
    Z_antithetic = -Z  # Trayectorias antitéticas

    # Cálculo de las trayectorias
    drift = (r - 0.5 * sigma ** 2) * tau
    diffusion = sigma * np.sqrt(tau)
    increments = drift + diffusion * Z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack((np.zeros((M2, 1)), log_paths))
    S = S0 * np.exp(log_paths)

    # Cálculo del promedio aritmético y payoff descontado
    I = np.mean(S[:, 1:], axis=1)
    payoff = np.maximum(I - K, 0)
    discounted1 = np.exp(-r * T) * payoff

    # Cálculo de las trayectorias antitéticas
    increments_a = drift + diffusion * Z_antithetic
    log_paths_a = np.cumsum(increments_a, axis=1)
    log_paths_a = np.hstack((np.zeros((M2, 1)), log_paths_a))
    S_a = S0 * np.exp(log_paths_a)

    # Cálculo del promedio aritmético y payoff descontado para antitéticas
    I_a = np.mean(S_a[:, 1:], axis=1)
    payoff_a = np.maximum(I_a - K, 0)
    discounted2 = np.exp(-r * T) * payoff_a

    # Promedio de ambas simulaciones
    total = 0.5 * (discounted1 + discounted2)

    return np.mean(total)


# **************************************************************** #
# Método de simulaciones de Montecarlo con variables antitéticas   #
# para opciones asiáticas con precio promedio y media geométrica   #
# **************************************************************** #
def MC_anthitetic_asianCall_AvgPrice_GEO(S0, K, T, r, sigma, N, M):
    tau = T / N
    M2 = M // 2

    # Generación M/2 trayectorias normales y sus trayectorias antitéticas
    Z = np.random.normal(size=(M2, N))
    Z_antithetic = -Z

    # Cálculo de las trayectorias
    drift = (r - 0.5 * sigma ** 2) * tau
    diffusion = sigma * np.sqrt(tau)
    increments = drift + diffusion * Z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack((np.zeros((M2, 1)), log_paths)) 
    S = S0 * np.exp(log_paths)

    # Cálculo del promedio geométrico y payoff descontado
    log_avg = np.mean(np.log(S[:, 1:]), axis=1)
    G = np.exp(log_avg)
    payoff = np.maximum(G - K, 0)
    discounted1 = np.exp(-r * T) * payoff

    # Cálculo de las trayectorias antitéticas
    increments_a = drift + diffusion * Z_antithetic
    log_paths_a = np.cumsum(increments_a, axis=1)
    log_paths_a = np.hstack((np.zeros((M2, 1)), log_paths_a))
    S_a = S0 * np.exp(log_paths_a)

    # Cálculo del promedio geométrico y payoff descontado para antitéticas
    log_avg_a = np.mean(np.log(S_a[:, 1:]), axis=1)
    G_a = np.exp(log_avg_a)
    payoff_a = np.maximum(G_a - K, 0)
    discounted2 = np.exp(-r * T) * payoff_a

    # Promedio de ambas simulaciones
    total = 0.5 * (discounted1 + discounted2)

    return np.mean(total)

# **************************************************************** #
# Fórmula cerrada de Kemna y Vorst de la ec. de B-S adaptada       #
# para opciones asiáticas con precio promedio y media geométrica   #
# **************************************************************** #
def EX_asianCall_AvgPrice_GEO(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r+(sigma**2)/6)*T/2)/(sigma*np.sqrt(T/3))
    d2 = (np.log(S0/K) + (r-(sigma**2)/2)*T/2)/(sigma*np.sqrt(T/3))
    return S0*np.exp(-(r + (sigma**2)/6)*T/2) * norm.cdf(d1) - K * np.exp(-r*T)* norm.cdf(d2)
