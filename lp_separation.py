"""
lp_separation.py

Separación lineal con Programación Lineal (PL) usando SciPy (HiGHS).

Qué hace:
1) Carga el dataset Breast Cancer (UCI) con ucimlrepo.
2) Preprocesa: convierte Diagnosis a clases {M,B} y forma A (malignas) y B (benignas).
3) Construye y resuelve el PL PRIMAL del proyecto:
      minimizar (1/m) * sum(y) + (1/p) * sum(z)
      s.a.  Aw + y >= e_A*beta + e_A
            Bw - z <= e_B*beta - e_B
            y >= 0, z >= 0
   con variables: w (n,), beta (esc), y (m,), z (p,).

   Se reescribe en forma A_ub @ v <= b_ub para linprog.

4) (Opcional) Resuelve el DUAL directamente desde los multiplicadores duales
   que entrega HiGHS y valida condiciones KKT de forma práctica.

5) Reporta: iteraciones, tiempo de CPU, valor óptimo, y checks KKT básicos.

6) Grafica en 2D los vectores (Aw + y) y (Bw - z) vs. índice (cumple “en 2D”).

Notas:
- Modelo conforme al enunciado del proyecto de Separación Lineal.
- Lenguaje simple y comentarios breves para fácil lectura.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from scipy.optimize import linprog
import matplotlib.pyplot as plt


# ------------- Utilidades de salida -----------------------------------------

@dataclass
class LPReport:
    x_opt: np.ndarray           # solución completa [w, beta, y, z]
    obj: float                  # valor óptimo
    iters: int                  # iteraciones reportadas por HiGHS
    cpu_time: float             # segundos
    status: str                 # mensaje de estado de SciPy
    success: bool               # True si óptimo encontrado
    slack_primal_A: np.ndarray  # slacks de restricciones de A (<= forma)
    slack_primal_B: np.ndarray  # slacks de restricciones de B (<= forma)
    dual_multipliers: np.ndarray  # lambdas de restricciones (A y B, ordenadas)


# ------------- Carga y preprocesamiento -------------------------------------

def load_breast_cancer() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Descarga el dataset Breast Cancer (Diagnostic) desde UCI.
    Devuelve:
      X: DataFrame (569 x 30) con variables numéricas.
      y: Serie con etiquetas 'M' o 'B'.
    """
    data = fetch_ucirepo(id=17)  # Breast Cancer Wisconsin (Diagnostic)
    X = data.data.features.copy()     # (569 x 30)
    y = data.data.targets["Diagnosis"].copy()  # 'M' o 'B'
    # Sin valores perdidos según el repo; no imputamos.
    return X, y


def standardize(X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Estandariza columnas a media 0 y varianza 1 (evita escalas muy distintas).
    Devuelve X_std y (mu, sigma) por si se desean usar después.
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0).replace(0, 1.0)  # evita división por cero
    X_std = (X - mu) / sigma
    return X_std, mu, sigma


def split_into_A_B(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separa los datos en matrices:
      A: renglones con Diagnosis == 'M' (malignas)
      B: renglones con Diagnosis == 'B' (benignas)
    Valores devueltos son np.ndarray (float64) para PL.
    """
    A = X[y == "M"].to_numpy(dtype=float)  # m x n
    B = X[y == "B"].to_numpy(dtype=float)  # p x n
    return A, B


# ------------- Construcción de PL (primal) ----------------------------------

def build_primal_lp(A: np.ndarray, B: np.ndarray):
    """
    Construye (c, A_ub, b_ub, bounds) para linprog.

    Variables v = [w (n), beta (1), y (m), z (p)]
      - w y beta: libres (sin acotar).
      - y, z: >= 0.

    Restricciones del enunciado (tras normalizar margen a 1):
      Aw + y >= e_A*beta + e_A
      Bw - z <= e_B*beta - e_B

    Convertimos a forma A_ub @ v <= b_ub:
      Para A:
        Aw - e_A*beta - y >= e_A
        -> (-A)w + (e_A)beta + (I_m)y <= -e_A
      Para B:
        Bw - e_B*beta - z <= -e_B
        -> ( B)w + (-e_B)beta + ( 0)y + (-I_p)z <= -e_B
    """
    m, n = A.shape
    p, n2 = B.shape
    assert n == n2

    # Tamaños
    n_w = n
    n_beta = 1
    n_y = m
    n_z = p
    N = n_w + n_beta + n_y + n_z

    # Bloques para A: (-A | e_A | I_m | 0)
    A_block = np.hstack([
        -A,                        # (-A) w
        np.ones((m, 1)),           # (e_A) beta
        np.eye(m),                 # + I_m * y
        np.zeros((m, p))           # 0 * z
    ])
    b_block = -np.ones(m)          # <= -e_A

    # Bloques para B: (B | -e_B | 0 | -I_p)
    B_block = np.hstack([
        B,                         # (B) w
        -np.ones((p, 1)),          # (-e_B) beta
        np.zeros((p, m)),          # 0 * y
        -np.eye(p)                 # -I_p * z
    ])
    b_block_B = -np.ones(p)        # <= -e_B

    A_ub = np.vstack([A_block, B_block])
    b_ub = np.concatenate([b_block, b_block_B])

    # Objetivo: min (1/m) e_A^T y + (1/p) e_B^T z
    c = np.zeros(N)
    c[n_w + n_beta : n_w + n_beta + n_y] = 1.0 / m   # coef para y
    c[-n_z:] = 1.0 / p                               # coef para z

    # Bounds: w y beta libres; y, z >= 0
    bounds = []
    bounds += [(None, None)] * n_w        # w
    bounds += [(None, None)]              # beta
    bounds += [(0.0, None)] * n_y         # y
    bounds += [(0.0, None)] * n_z         # z

    return c, A_ub, b_ub, bounds, (m, n, p)


def solve_primal(c, A_ub, b_ub, bounds) -> Tuple[np.ndarray, dict]:
    """
    Resuelve el primal con HiGHS (linprog). Devuelve solución y metadata.
    """
    t0 = time.perf_counter()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    t1 = time.perf_counter()
    meta = {
        "cpu_time": t1 - t0,
        "iters": res.nit if hasattr(res, "nit") else None,
        "status": res.message,
        "success": bool(res.success),
        "result": res,
    }
    return res.x, meta


# ------------- Validaciones y reportes --------------------------------------

def kkt_checks(c, A_ub, b_ub, x_opt, res) -> Tuple[np.ndarray, np.ndarray]:
    """
    Checks KKT básicos (prácticos) a partir de la salida de HiGHS.

    - Factibilidad primal: A_ub @ x <= b_ub
    - Multiplicadores (dual): lambdas >= 0 (HiGHS entrega .ineqlin)
    - Holgura complementaria: lambda_i * slack_i ≈ 0

    Devuelve:
      (slack, lambda)
    """
    # Slack primal
    slack = b_ub - A_ub @ x_opt

    # Multiplicadores duales de inecuaciones (HiGHS)
    # SciPy guarda en res.ineqlin['marginals'] o res.dual_ineqlin (según versión)
    if hasattr(res, "ineqlin") and "marginals" in res.ineqlin:
        lambdas = np.asarray(res.ineqlin["marginals"])
    elif hasattr(res, "lambda_ineqlin"):  # fallback en algunas versiones
        lambdas = np.asarray(res.lambda_ineqlin)
    else:
        # Si no está disponible, devolvemos NaN para indicar “no disponible”.
        lambdas = np.full_like(slack, np.nan, dtype=float)

    return slack, lambdas


def build_report(A: np.ndarray, B: np.ndarray, c, A_ub, b_ub, bounds, x_opt, meta) -> LPReport:
    """
    Arma un objeto con todo lo útil: x_opt, obj, iters, tiempo, slacks y lambdas.
    """
    res = meta["result"]
    slack, lambdas = kkt_checks(c, A_ub, b_ub, x_opt, res)

    m, n = A.shape
    p = B.shape[0]

    # Particiona la solución para comodidad
    w = x_opt[:n]
    beta = x_opt[n]
    y = x_opt[n+1 : n+1+m]
    z = x_opt[-p:]

    # Reconstruye los vectores pedidos por el enunciado
    Aw_plus_y = A @ w + y - np.ones(m) * (beta)   # = (Aw + y) - e_A*beta
    # Nota: para graficar el “Aw + y” como tal, usamos: A@w + y
    Bw_minus_z = B @ w - z - np.ones(p) * (beta)  # = (Bw - z) - e_B*beta

    # Guarda slacks separados por bloque
    slack_A = slack[:m]
    slack_B = slack[m:]

    return LPReport(
        x_opt=x_opt,
        obj=float(res.fun) if meta["success"] else np.inf,
        iters=int(meta["iters"]) if meta["iters"] is not None else -1,
        cpu_time=float(meta["cpu_time"]),
        status=str(meta["status"]),
        success=meta["success"],
        slack_primal_A=slack_A,
        slack_primal_B=slack_B,
        dual_multipliers=lambdas
    )


# ------------- Gráficas solicitadas -----------------------------------------

def plot_vectors(A: np.ndarray, B: np.ndarray, w: np.ndarray, beta: float, y: np.ndarray, z: np.ndarray):
    """
    Grafica en 2D (índice vs valor) los vectores:
      - Aw + y
      - Bw - z
    Esto cumple con “graficar en dos dimensiones” del enunciado.
    """
    m = A.shape[0]
    p = B.shape[0]

    Aw_plus_y = A @ w + y
    Bw_minus_z = B @ w - z

    plt.figure()
    plt.title("Vector Aw + y (índice vs valor)")
    plt.plot(np.arange(m), Aw_plus_y, marker="o", linestyle="None")
    plt.xlabel("Índice en A")
    plt.ylabel("Aw + y")
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.title("Vector Bw - z (índice vs valor)")
    plt.plot(np.arange(p), Bw_minus_z, marker="o", linestyle="None")
    plt.xlabel("Índice en B")
    plt.ylabel("Bw - z")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


# ------------- Main ----------------------------------------------------------

def main():
    # 1) Carga y preprocesa
    X, y = load_breast_cancer()
    X_std, mu, sigma = standardize(X)  # recomendado para estabilidad numérica
    A, B = split_into_A_B(X_std, y)

    # 2) Construye PL
    c, A_ub, b_ub, bounds, sizes = build_primal_lp(A, B)

    # 3) Resuelve
    x_opt, meta = solve_primal(c, A_ub, b_ub, bounds)
    res = meta["result"]

    # 4) Reporte + KKT
    report = build_report(A, B, c, A_ub, b_ub, bounds, x_opt, meta)

    # 5) Imprime métricas
    print("\n=== Resultados PRIMAL (HiGHS) ===")
    print(f"Éxito:        {report.success}")
    print(f"Estatus:      {report.status}")
    print(f"Iteraciones:  {report.iters}")
    print(f"CPU (s):      {report.cpu_time:.6f}")
    print(f"Objetivo:     {report.obj:.6f}")

    # Particiona solución para graficar
    n = X_std.shape[1]
    m = A.shape[0]
    p = B.shape[0]
    w = report.x_opt[:n]
    beta = report.x_opt[n]
    y_slack = report.x_opt[n+1 : n+1+m]
    z_slack = report.x_opt[-p:]

    # 6) Chequeos KKT simples
    slack = np.concatenate([report.slack_primal_A, report.slack_primal_B])
    lambdas = report.dual_multipliers
    if not np.any(np.isnan(lambdas)):
        compl = lambdas * slack
        print("\n--- KKT (práctico) ---")
        print(f"Min slack primal: {slack.min():.3e} (debería ser >= 0)")
        print(f"Min lambda dual:  {lambdas.min():.3e} (debería ser >= 0)")
        print(f"||lambda ∘ slack||∞: {np.max(np.abs(compl)):.3e} (≈0 sugiere holgura complementaria)")
    else:
        print("\n--- KKT ---")
        print("Multiplicadores duales no disponibles en esta versión de SciPy/HiGHS.")

    # 7) Gráficas solicitadas
    plot_vectors(A, B, w, beta, y_slack, z_slack)


if __name__ == "__main__":
    main()
