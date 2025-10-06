"""
  (a) PRIMAL (con holguras)
  (b) DUAL explícito (multiplicadores)

  - Nº de iteraciones (primal/dual)
  - Tiempo de CPU (s)
  - Valor óptimo (primal y dual) + brecha |primal − dual|
  - Validación KKT: min_slack, ||λ∘slack||_inf, ||c + A^T λ||_inf
  - Figuras: Aw+y, Bw−z, PCA 2D

Requisitos:
  pip install numpy pandas scipy matplotlib scikit-learn ucimlrepo
"""

from __future__ import annotations
import os, time, json
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo
from scipy.optimize import linprog


# ======================== utilidades I/O ========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ======================== datos ========================

def load_breast_cancer() -> Tuple[pd.DataFrame, pd.Series]:
    """Breast Cancer Wisconsin (Diagnostic)."""
    data = fetch_ucirepo(id=17)
    X = data.data.features.copy()
    y = data.data.targets["Diagnosis"].copy()  # 'M' o 'B'
    return X, y

def standardize(X: pd.DataFrame) -> pd.DataFrame:
    """Media 0, varianza 1 (estable para Simplex)."""
    mu = X.mean(0)
    sigma = X.std(0).replace(0, 1.0)
    return (X - mu) / sigma

def split_A_B(Xs: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """A: malignos (M); B: benignos (B)."""
    A = Xs[y == "M"].to_numpy(float)
    B = Xs[y == "B"].to_numpy(float)
    return A, B


# ======================== PRIMAL (forma estándar p/ Simplex) ========================
# Modelo del proyecto (margen 1):
#   min (1/m) 1^T y + (1/p) 1^T z
#   s.a.  Aw + y >= e_A β + e_A
#         Bw - z <= e_B β - e_B
#         y,z >= 0 ; w,β libres
#
# Para Simplex con A_eq x = b y x >= 0, hacemos "variable splitting":
#   β = β_pos - β_neg,  w = w_pos - w_neg,  y,z >= 0
# y convertimos las desigualdades a IGUALDADES con slacks r,s (>=0).

def build_primal_eq(A: np.ndarray, B: np.ndarray):
    m, n = A.shape
    p = B.shape[0]

    # Bloque A (m ecuaciones): -e_A β+ + e_A β- + A w+ − A w- + I y − I r = e_A
    M_A = np.hstack([
        -np.ones((m, 1)),   # β_pos
        +np.ones((m, 1)),   # β_neg
        +A,                 # w_pos
        -A,                 # w_neg
        np.eye(m),          # y
        np.zeros((m, p)),   # z
        -np.eye(m),         # r (slack A)
        np.zeros((m, p)),   # s (slack B)
    ])
    b_A = np.ones(m)

    # Bloque B (p ecuaciones): -e_B β+ + e_B β- + B w+ − B w- − I z + I s = −e_B
    M_B = np.hstack([
        -np.ones((p, 1)),
        +np.ones((p, 1)),
        +B,
        -B,
        np.zeros((p, m)),
        -np.eye(p),
        np.zeros((p, m)),
        +np.eye(p),
    ])
    b_B = -np.ones(p)

    M = np.vstack([M_A, M_B])
    b = np.concatenate([b_A, b_B])

    # Costos: sólo y y z
    N = 2 + 2*n + m + p + m + p
    c = np.zeros(N)
    c[2 + 2*n : 2 + 2*n + m] = 1.0/m
    c[2 + 2*n + m : 2 + 2*n + m + p] = 1.0/p

    bounds = [(0, None)] * N
    sizes = (m, n, p)
    return c, M, b, bounds, sizes

# Además, armamos la versión en INECUACIONES (para KKT con λ=[u;q] y variables [w,β,y,z]):

def build_primal_ub(A: np.ndarray, B: np.ndarray):
    """
    Devuelve (c, A_ub, b_ub, bounds) para variables ordenadas como:
      v = [ w (n), beta (1), y (m), z (p) ]   con y,z >= 0; w,beta libres.
    Desigualdades en forma <= conforme al enunciado transformado:
      (-A) w + 1*beta + 1*y <= -1        (m filas)
      ( B) w - 1*beta - 1*z <= -1        (p filas)
    """
    m, n = A.shape
    p = B.shape[0]

    A_block = np.hstack([ -A, np.ones((m,1)), np.eye(m), np.zeros((m,p)) ])
    B_block = np.hstack([  B, -np.ones((p,1)), np.zeros((p,m)), -np.eye(p) ])

    A_ub = np.vstack([A_block, B_block])
    b_ub = -np.ones(m + p)

    c = np.zeros(n + 1 + m + p)
    c[n+1:n+1+m] = 1.0/m
    c[n+1+m:]     = 1.0/p

    bounds = [(None, None)]*n + [(None, None)] + [(0, None)]*m + [(0, None)]*p
    return c, A_ub, b_ub, bounds


# ======================== DUAL (forma estándar p/ Simplex) ========================
# Variables: u (m), q (p), f (m), g (p) >= 0
# Igualdades:
#   e_A^T u − e_B^T q = 0
#   A^T u − B^T q = 0
#   u + f = e_A/m
#   q + g = e_B/p
# Objetivo (min, para linprog):  min  -(1^T u + 1^T q)  (→ equivale a max 1^T u + 1^T q con signo opuesto)

def build_dual_eq(A: np.ndarray, B: np.ndarray):
    m, n = A.shape
    p = B.shape[0]
    eA = np.ones((m,1)); eB = np.ones((p,1))

    M1 = np.hstack([ eA.T, -eB.T, np.zeros((1, m+p)) ])      # 1×(m+p+m+p)
    M2 = np.hstack([ A.T,  -B.T,  np.zeros((n, m+p)) ])      # n×(...)
    M3 = np.hstack([ np.eye(m), np.zeros((m,p)),  np.eye(m), np.zeros((m,p)) ])  # u+f
    M4 = np.hstack([ np.zeros((p,m)), np.eye(p), np.zeros((p,m)), np.eye(p) ])   # q+g
    M = np.vstack([M1, M2, M3, M4])

    b = np.concatenate([ np.zeros(1+n), (eA/m).ravel(), (eB/p).ravel() ])
    c = np.concatenate([ -np.ones(m), -np.ones(p), np.zeros(m+p) ])  # min -sum(u)-sum(q)

    bounds = [(0, None)] * (m + p + m + p)
    return c, M, b, bounds


# ======================== solver Simplex (A_eq x = b, x>=0) ========================

def simplex_eq(c, M, b, bounds, maxiter: int = 10000):
    t0 = time.perf_counter()
    res = linprog(c, A_eq=M, b_eq=b, bounds=bounds, method="simplex", options={"maxiter": maxiter})
    t1 = time.perf_counter()
    meta = {
        "success": bool(res.success),
        "status": res.message,
        "obj": float(res.fun) if res.success else np.inf,
        "iters": getattr(res, "nit", None),
        "cpu": t1 - t0,
        "res": res,
    }
    return res.x, meta


# ======================== KKT (coherente con el modelo en <=) ========================

def kkt_checks_with_uq(w: np.ndarray, beta: float, y: np.ndarray, z: np.ndarray,
                       A: np.ndarray, B: np.ndarray, u: np.ndarray, q: np.ndarray):
    """
    Calcula KKT usando la forma en inecuaciones:
      A_ub v <= b_ub  con v = [w, beta, y, z]
      λ = [u; q] >= 0
    Devuelve dict con: min_slack, comp_inf, station_inf
    """
    # Construimos A_ub, b_ub y c para v = [w,beta,y,z]
    c_v, A_ub, b_ub, _ = build_primal_ub(A, B)
    v = np.concatenate([w, [beta], y, z])
    slack = b_ub - A_ub @ v
    lam = np.concatenate([u, q])

    min_slack = float(slack.min())
    comp_inf = float(np.max(np.abs(lam * slack)))
    station_inf = float(np.max(np.abs(c_v + A_ub.T @ lam)))
    return {"min_slack": min_slack, "comp_inf": comp_inf, "station_inf": station_inf}


# ======================== gráficas ========================

def plot_Aw_plus_y(A, w, y, path_png):
    vals = A @ w + y
    plt.figure()
    plt.title("Aw + y")
    plt.plot(np.arange(len(vals)), vals, marker="o", linestyle="None")
    plt.xlabel("Índice en A"); plt.ylabel("Aw + y"); plt.grid(True); plt.tight_layout()
    plt.savefig(path_png); plt.close()

def plot_Bw_minus_z(B, w, z, path_png):
    vals = B @ w - z
    plt.figure()
    plt.title("Bw - z")
    plt.plot(np.arange(len(vals)), vals, marker="o", linestyle="None")
    plt.xlabel("Índice en B"); plt.ylabel("Bw - z"); plt.grid(True); plt.tight_layout()
    plt.savefig(path_png); plt.close()

def plot_pca_hyperplane(Xs: pd.DataFrame, y: pd.Series, w: np.ndarray, beta: float, path_png: str):
    """
    PCA a 2D (PC1, PC2) y recta del hiperplano proyectada. Es ilustrativa.
    """
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(Xs.to_numpy())
    w2 = pca.components_ @ w  # proyección de w

    x1 = np.linspace(X2[:,0].min(), X2[:,0].max(), 400)
    if abs(w2[1]) < 1e-10:
        xline = None
    else:
        xline = (beta - w2[0]*x1) / w2[1]

    plt.figure()
    maskM = (y.values == "M")
    plt.scatter(X2[~maskM,0], X2[~maskM,1], s=14, label="B")
    plt.scatter(X2[ maskM,0], X2[ maskM,1], s=14, label="M")
    if xline is None:
        # Vertical aprox (w2[1]≈0)
        x0 = beta / (w2[0] if abs(w2[0])>1e-12 else 1.0)
        plt.axvline(x=x0, linestyle="--", label="Hiperplano (aprox)")
    else:
        plt.plot(x1, xline, "--", label="Hiperplano (aprox)")
    plt.title("PCA 2D con hiperplano proyectado")
    plt.legend(); plt.tight_layout(); plt.savefig(path_png); plt.close()


# ======================== pipeline principal ========================

def run_simplex(outdir: str = "outputs_simplex") -> Dict[str, Any]:
    """
    Ejecuta TODO con Simplex (primal y dual) y REGRESA un dict con:
      - iteraciones, cpu, objetivo (primal/dual)
      - brecha de dualidad |primal − dual|
      - KKT (min_slack, comp_inf, station_inf)
      - rutas de: Aw+y, Bw−z, pca_hyperplane
    """
    ensure_dir(outdir)

    # 1) Datos y estandarización
    X, y = load_breast_cancer()
    Xs = standardize(X)          # recomendado
    A, B = split_A_B(Xs, y)

    # 2) PRIMAL (eq) → Simplex
    cP, MP, bP, bndP, sizes = build_primal_eq(A, B)
    xP, metaP = simplex_eq(cP, MP, bP, bndP)
    m, n, p = sizes

    # Particionar solución primal
    beta = xP[0] - xP[1]
    w    = xP[2:2+n] - xP[2+n:2+2*n]
    y_sl = xP[2+2*n : 2+2*n + m]
    z_sl = xP[2+2*n + m : 2+2*n + m + p]

    # 3) DUAL (eq) → Simplex
    cD, MD, bD, bndD = build_dual_eq(A, B)
    xD, metaD = simplex_eq(cD, MD, bD, bndD)

    # u, q son los primeros m y p elementos, respectivamente
    u = xD[:m]
    q = xD[m:m+p]

    # Objetivo dual para reportar en “max” (cambiamos signo)
    obj_dual_report = - metaD["obj"]

    # 4) KKT coherente con inecuaciones originales (<=)
    kkt = kkt_checks_with_uq(w, beta, y_sl, z_sl, A, B, u, q)

    # 5) Brecha de dualidad
    gap = abs(metaP["obj"] - obj_dual_report) if (metaP["success"] and metaD["success"]) else None

    # 6) Gráficas
    fig1 = os.path.join(outdir, "Aw_plus_y_simplex.png")
    fig2 = os.path.join(outdir, "Bw_minus_z_simplex.png")
    fig3 = os.path.join(outdir, "pca_hyperplane_simplex.png")
    plot_Aw_plus_y(A, w, y_sl, fig1)
    plot_Bw_minus_z(B, w, z_sl, fig2)
    plot_pca_hyperplane(Xs, y, w, beta, fig3)

    # 7) Empaquetar resultados
    results = {
        "solver": "simplex",
        "sizes": {"m": int(m), "p": int(p), "n": int(n)},
        "primal": {
            "success": metaP["success"], "status": metaP["status"],
            "iterations": metaP["iters"], "cpu_seconds": metaP["cpu"],
            "objective": metaP["obj"],
        },
        "dual": {
            "success": metaD["success"], "status": metaD["status"],
            "iterations": metaD["iters"], "cpu_seconds": metaD["cpu"],
            "objective_max_form": obj_dual_report,  # ¡en forma max!
        },
        "duality_gap_abs": gap,
        "KKT": kkt,
        "figures": {
            "Aw_plus_y": fig1,
            "Bw_minus_z": fig2,
            "pca_2d": fig3,
        },
        "hyperplane": {
            "beta": float(beta),
            "w_norm": float(np.linalg.norm(w)),
        }
    }
    save_json(os.path.join(outdir, "results_simplex.json"), results)
    return results


# ======================== CLI ========================

if __name__ == "__main__":
    out = run_simplex()
    print("\n=== SIMPLEX — Resumen ===")
    print(f"Datos: m={out['sizes']['m']} (A), p={out['sizes']['p']} (B), n={out['sizes']['n']} (features)")
    print(f"PRIMAL  -> iter={out['primal']['iterations']}, cpu={out['primal']['cpu_seconds']:.6f}s, "
          f"obj={out['primal']['objective']:.6f}, success={out['primal']['success']}")
    print(f"DUAL    -> iter={out['dual']['iterations']}, cpu={out['dual']['cpu_seconds']:.6f}s, "
          f"obj(max)={out['dual']['objective_max_form']:.6f}, success={out['dual']['success']}")
    print(f"Gap |primal - dual| = {out['duality_gap_abs']}")
    print(f"KKT -> min_slack={out['KKT']['min_slack']:.3e}, "
          f"||λ∘slack||_inf={out['KKT']['comp_inf']:.3e}, "
          f"||c + A^T λ||_inf={out['KKT']['station_inf']:.3e}")
    print("Figuras guardadas:")
    for k, v in out["figures"].items():
        print(" -", k, ":", v)
