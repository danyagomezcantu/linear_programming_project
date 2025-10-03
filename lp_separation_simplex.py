"""
lp_separation_simplex.py

Separación lineal por Programación Lineal (PL) usando EXCLUSIVAMENTE
linprog(method="simplex") para:
  (a) PRIMAL (con holguras)
  (b) DUAL explícito (variables lambda)

El script:
- Carga Breast Cancer (UCI) y estandariza.
- Construye y resuelve PRIMAL y DUAL con SIMPLEX.
- Calcula y DEVUELVE métricas pedidas:
    * iteraciones (primal/dual)
    * tiempo de CPU (primal/dual)
    * valor óptimo (primal y dual) y brecha |primal - dual|
    * validación KKT (min slack, complementariedad, estacionariedad)
- Genera y guarda las gráficas solicitadas: Aw + y y Bw - z.
- Regresa un dict con todo (rutas de imágenes incluidas).

Modelo conforme al enunciado del proyecto (margen normalizado a 1,
variables de holgura, y graficación de Aw+y y Bw-z).
"""

from __future__ import annotations
import os, time, json
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from ucimlrepo import fetch_ucirepo


# -------------------- utilidades sencillas --------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -------------------- Obtención de Datos --------------------


def load_breast_cancer() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carga el dataset Breast Cancer Wisconsin (Diagnostic) desde UCI.
    Devuelve:
      X: DataFrame (569x30) con variables numéricas.
      y: Serie con etiquetas 'M' (maligno) y 'B' (benigno).
    """
    data = fetch_ucirepo(id=17)
    X = data.data.features.copy()
    y = data.data.targets["Diagnosis"].copy()
    return X, y


def standardize(X: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza cada columna a media 0 y varianza 1.
    Esto ayuda a la estabilidad numérica del PL.
    """
    mu = X.mean(0)
    sigma = X.std(0).replace(0, 1.0)
    return (X - mu) / sigma


def split_A_B(Xs: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separa matrices A (clase 'M') y B (clase 'B') como arreglos float.
    """
    A = Xs[y == "M"].to_numpy(float)
    B = Xs[y == "B"].to_numpy(float)
    return A, B


# ------------------------- PROBLEMA PRIMAL -------------------------
#
#   Min (e_A/m)^T @ y + (e_B/p)^T @ z
#   S.a. Aw + y >= e_A*beta + e_A
#        Bw - z <= e_B*beta - e_B
#        y >= 0, z >= 0; w y alfa libres
# Para linprog usaremos la forma canónica (max y <=):
#   (-A)w + (e_A)beta + I*y + 0*z <= -e_A
#   ( B)w + (-e_B)beta + 0*y + (-I)z <= -e_B


def build_primal(A: np.ndarray, B: np.ndarray):
    m, n = A.shape
    p = B.shape[0]

    A_block = np.hstack([-A, np.ones((m, 1)), np.eye(m), np.zeros((m, p))])
    bA = -np.ones(m)

    B_block = np.hstack([B, -np.ones((p, 1)), np.zeros((p, m)), -np.eye(p)])
    bB = -np.ones(p)

    A_ub = np.vstack([A_block, B_block])
    b_ub = np.concatenate([bA, bB])

    # c: sólo y, z aparecen en el objetivo
    N = n + 1 + m + p
    c = np.zeros(N)
    c[n + 1 : n + 1 + m] = 1.0 / m
    c[-p:] = 1.0 / p

    # w y beta libres; y,z >= 0
    bounds = [(-float("inf"), float("inf"))] * (n + 1) + [(0, float("inf"))] * (m + p)
    sizes = (m, n, p)
    return c, A_ub, b_ub, bounds, sizes


def solve_primal_simplex(c, A_ub, b_ub, bounds):
    """
    Resuelve el primal con simplex clásico de SciPy.
    Devuelve (x_opt, meta, res) donde 'meta' contiene iteraciones, cpu, etc.
    """
    t0 = time.perf_counter()
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="simplex",
        options={"maxiter": 5000},
    )
    t1 = time.perf_counter()
    meta = {
        "success": bool(res.success),
        "status": res.message,
        "obj": float(res.fun) if res.success else np.inf,
        "iters": getattr(res, "nit", None),
        "cpu": t1 - t0,
    }
    return res.x, meta, res


# -------------------- DUAL explícito (forma) --------------------
# Variables duales: u en R^m, q en R^p
# Máx  (e_A)^T @ u + (e_B)^T @ q    <=>  Min  1^T λ_A + 1^T λ_B  (cambiando el signo)
# s.a.
#   0 ≤ u ≤ (e_A)/m,   0 ≤ q ≤ (e_B)/p
#    -A^T λ_A +  B^T λ_B ≤ 0     (por w_+)
#    A^T λ_A -  B^T λ_B ≤ 0     (por w_-)
#    e_A^T λ_A -  e_B^T λ_B ≤ 0 (por beta_+)
#   -e_A^T λ_A +  e_B^T λ_B ≤ 0 (por beta_-)
# Nota: en óptimo, e_A^T λ_A = e_B^T λ_B.


def build_dual(A: np.ndarray, B: np.ndarray):
    m, n = A.shape
    p = B.shape[0]
    # Restricciones (≤) sobre [λ_A; λ_B]
    C1 = np.hstack([-A.T, B.T])  # w^+
    C2 = np.hstack([A.T, -B.T])  # w^-
    eA = np.ones((m, 1))
    eB = np.ones((p, 1))
    C3 = np.hstack([eA.T, -eB.T])  # beta^+
    C4 = np.hstack([-eA.T, eB.T])  # beta^-
    A_ub = np.vstack([C1, C2, C3, C4])
    b_ub = np.zeros(2 * n + 2)

    # Objetivo de minimización (para usar linprog):
    # min 1^T λ_A + 1^T λ_B  y luego negamos para reportar la forma "max -sum λ"
    c = np.concatenate([np.ones(m), np.ones(p)])

    # Bounds: 0 ≤ λ_A ≤ 1/m,  0 ≤ λ_B ≤ 1/p
    bounds = [(0, 1.0 / m)] * m + [(0, 1.0 / p)] * p
    return c, A_ub, b_ub, bounds


def solve_dual_simplex(A: np.ndarray, B: np.ndarray):
    """
    Resuelve el dual explícito con simplex (min versión) y devuelve:
      - lbd_opt: lambdas óptimas concatenadas [λ_A; λ_B]
      - meta_dual: dict con éxito, iters, cpu y obj_dual (en forma "max")
      - res: objeto de SciPy
    """
    c, A_ub, b_ub, bounds = build_dual(A, B)
    t0 = time.perf_counter()
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="simplex",
        options={"maxiter": 5000},
    )
    t1 = time.perf_counter()
    # pasamos de "min sum λ" a "max -sum λ"
    obj_dual = float(-(res.fun)) if res.success else -np.inf
    meta = {
        "success": bool(res.success),
        "status": res.message,
        "obj_dual": obj_dual,
        "iters": getattr(res, "nit", None),
        "cpu": t1 - t0,
    }
    return res.x, meta, res


# -------------------- KKT (con lambdas del DUAL explícito) --------------------


def kkt_from_primal_dual(
    c: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    x_opt: np.ndarray,
    lam_opt: np.ndarray,
) -> dict:
    """
    KKT prácticos:
      - Factibilidad primal: A_ub x <= b_ub  (medimos min slack)
      - λ >= 0 (lo asegura el dual por bounds)
      - Complementariedad: λ_i * slack_i ~ 0
      - Estacionariedad: || c + A_ub^T λ ||_inf ~ 0
    """
    slack = b_ub - A_ub @ x_opt
    min_slack = float(slack.min())
    comp_inf = float(np.max(np.abs(lam_opt * slack)))
    station_inf = float(np.max(np.abs(c + A_ub.T @ lam_opt)))
    return {"min_slack": min_slack, "comp_inf": comp_inf, "station_inf": station_inf}


# -------------------- gráficas pedidas --------------------


def plot_Aw_plus_y(A: np.ndarray, w: np.ndarray, y: np.ndarray, path_png: str) -> None:
    vals = A @ w + y
    plt.figure()
    plt.title("Aw + y")
    plt.plot(np.arange(len(vals)), vals, marker="o", linestyle="None")
    plt.xlabel("Índice en A")
    plt.ylabel("Aw + y")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()


def plot_Bw_minus_z(B: np.ndarray, w: np.ndarray, z: np.ndarray, path_png: str) -> None:
    vals = B @ w - z
    plt.figure()
    plt.title("Bw - z")
    plt.plot(np.arange(len(vals)), vals, marker="o", linestyle="None")
    plt.xlabel("Índice en B")
    plt.ylabel("Bw - z")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()


# -------------------- experimento principal (lo que te regresa) --------------------


def run_simplex_experiment(outdir: str = "outputs_simplex") -> Dict[str, Any]:
    """
    Ejecuta TODO con SIMPLEX (primal y dual) y REGRESA un dict con:
      - iteraciones, cpu y objetivo (primal y dual)
      - brecha de dualidad
      - métricas KKT (min_slack, comp_inf, station_inf)
      - rutas de las figuras Aw+y y Bw-z

    También guarda 'results_simplex.json' y las imágenes en 'outdir'.
    """
    ensure_dir(outdir)

    # 1) Datos
    X, y = load_breast_cancer()
    Xs = standardize(X)
    A, B = split_A_B(Xs, y)

    # 2) PRIMAL
    c, A_ub, b_ub, bounds, sizes = build_primal(A, B)
    x_opt, meta_p, res_p = solve_primal_simplex(c, A_ub, b_ub, bounds)

    # Particiona solución para graficar
    m, n, p = sizes
    w = x_opt[:n]
    beta = x_opt[n]
    y_slack = x_opt[n + 1 : n + 1 + m]
    z_slack = x_opt[-p:]

    # 3) DUAL (explícito)
    lam_opt, meta_d, res_d = solve_dual_simplex(A, B)

    # 4) KKT (usando los lambdas del dual explícito)
    kkt = kkt_from_primal_dual(c, A_ub, b_ub, x_opt, lam_opt)

    # 5) Brecha de dualidad
    gap = (
        abs(meta_p["obj"] - meta_d["obj_dual"])
        if (meta_p["success"] and meta_d["success"])
        else None
    )

    # 6) Gráficas pedidas
    fig1 = os.path.join(outdir, "Aw_plus_y_simplex.png")
    fig2 = os.path.join(outdir, "Bw_minus_z_simplex.png")
    plot_Aw_plus_y(A, w, y_slack, fig1)
    plot_Bw_minus_z(B, w, z_slack, fig2)

    # 7) Empaquetado de resultados
    results = {
        "solver": "simplex",
        "primal": {
            "success": meta_p["success"],
            "status": meta_p["status"],
            "iterations": meta_p["iters"],
            "cpu_seconds": meta_p["cpu"],
            "objective": meta_p["obj"],
        },
        "dual": {
            "success": meta_d["success"],
            "status": meta_d["status"],
            "iterations": meta_d["iters"],
            "cpu_seconds": meta_d["cpu"],
            "objective": meta_d["obj_dual"],
        },
        "duality_gap_abs": gap,
        "KKT": kkt,
        "figures": {"Aw_plus_y": fig1, "Bw_minus_z": fig2},
        "sizes": {"m": int(m), "p": int(p), "n": int(n)},
    }

    # 8) Guardar JSON
    save_json(os.path.join(outdir, "results_simplex.json"), results)
    return results


# -------------------- CLI simple --------------------

if __name__ == "__main__":
    out = run_simplex_experiment()
    # Impresión breve y clara
    print("\n=== SIMPLEX — Resumen ===")
    print(
        f"PRIMAL:  iters={out['primal']['iterations']}, cpu={out['primal']['cpu_seconds']:.6f}s, obj={out['primal']['objective']:.6f}, success={out['primal']['success']}"
    )
    print(
        f"DUAL:    iters={out['dual']['iterations']}, cpu={out['dual']['cpu_seconds']:.6f}s, obj={out['dual']['objective']:.6f}, success={out['dual']['success']}"
    )
    print(f"Gap |primal - dual| = {out['duality_gap_abs']}")
    print(
        f"KKT: min_slack={out['KKT']['min_slack']:.3e}, ||λ∘slack||_inf={out['KKT']['comp_inf']:.3e}, ||c + A^T λ||_inf={out['KKT']['station_inf']:.3e}"
    )
    print("Figuras guardadas:")
    print(" -", out["figures"]["Aw_plus_y"])
    print(" -", out["figures"]["Bw_minus_z"])
