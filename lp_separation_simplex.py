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
from sklearn.decomposition import PCA

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


def split_A_B(Xs: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separa matrices A (clase 'M') y B (clase 'B') como arreglos float.
    """
    A = Xs[y == "M"].to_numpy(float)
    B = Xs[y == "B"].to_numpy(float)
    return A, B


# ------------------------- PROBLEMA PRIMAL -------------------------
#
#   Min   (e_A/m)^T @ y + (e_B/p)^T @ z
#   S.a.  Aw + y >= e_A*beta + e_A
#         Bw - z <= e_B*beta - e_B
#         y >= 0, z >= 0; w y alfa libres
#
# En forma estándar: Sean beta_pos, beta_neg en R^1; w_pos, w_neg en R^n; r en R^m; s en R^p

#   Min     0*beta_pos + 0*beta_neg + 0^T@w_pos + 0^T@w_neg + (e_A/m)^T@y + (e_B/p)^T@z + 0^T@r + 0^T@s
#   S.a.  - e_A*beta_pos + e_A*beta_neg + A@w_pos - A@w_neg + I_m@y + 0_mxp@z - I_m@r + 0_mxp@s =  e_A
#         - e_B*beta_pos + e_B*beta_neg + B@w_pos - B@w_neg + 0_pxm@y - I_p@z + 0_pxm@z + I_p@s = -e_B
#           beta_pos, beta_neg, w_pos, w_neg, y, z, r, s >= 0


def construir_primal(A: np.ndarray, B: np.ndarray):
    m, n = A.shape
    p = B.shape[0]

    M_A = np.hstack(
        [
            -np.ones((m, 1)),
            np.ones((m, 1)),
            A,
            -A,
            np.eye(m),
            np.zeros((m, p)),
            -np.eye(m),
            np.zeros((m, p)),
        ]
    )
    b_A = np.ones(m)

    M_B = np.hstack(
        [
            -np.ones((p, 1)),
            np.ones((p, 1)),
            B,
            -B,
            np.zeros((p, m)),
            -np.eye(p),
            np.zeros((p, m)),
            np.eye(p),
        ]
    )
    b_B = -np.ones(p)

    M = np.vstack([M_A, M_B])
    b = np.concatenate([b_A, b_B])

    # c: sólo y, z aparecen en el objetivo
    N = 1 + 1 + n + n + m + p + m + p
    c = np.zeros(N)
    c[2 * (n + 1) : 2 * (n + 1) + m] = 1.0 / m
    c[2 * (n + 1) + m : 2 * (n + 1) + m + p] = 1.0 / p

    # w y beta libres; y,z >= 0
    bounds = [(0, float("inf"))] * (N)
    sizes = (m, n, p)
    return c, M, b, bounds, sizes


# ------------------------- PROBLEMA DUAL -------------------------
#
# Variables duales: u en R^m, q en R^p
#
#   Max   (e_A)^T @ u + (e_B)^T @ q
#   S.a   (e_A)^T @ u = (e_B)^T @ q
#         A^T @ u = B^T @ q
#         0 ≤ u ≤ (e_A)/m
#         0 ≤ q ≤ (e_B)/p
#
# En forma estándar: Sean f en R^m, g en R^p
#
#   Min   - (e_A)^T @ u - (e_B)^T @ q + 0^T @ f + 0^T @ g
#   S.a.    (e_A)^T @ u - (e_B)^T @ q + 0^T @ f + 0^T @ g = 0
#               A^T @ u - B^T @ q + 0_nxm @ f + 0_nxp @ g = 0
#               I_m @ u + 0_mxp @ q + I_m @ f + 0_mxp @ g = e_A/m
#               0_pxm @ u + I_m @ q + 0_pxm @ f + I_m @ g = e_B/p
#           u, q, f, g >= 0


def construir_dual(A: np.ndarray, B: np.ndarray):
    m, n = A.shape
    p = B.shape[0]

    e_A = np.ones((m, 1))
    e_B = np.ones((p, 1))

    M_1 = np.hstack([e_A.T, -e_B.T, np.zeros((1, m + p))])
    M_2 = np.hstack([A.T, -B.T, np.zeros((n, m + p))])
    M_3 = np.hstack([np.eye(m), np.zeros((m, p)), np.eye(m), np.zeros((m, p))])
    M_4 = np.hstack([np.zeros((p, m)), np.eye(p), np.zeros((p, m)), np.eye(p)])

    M = np.vstack([M_1, M_2, M_3, M_4])
    b = np.concatenate([np.zeros((1 + n, 1)), e_A / m, e_B / p])

    c = np.concatenate([-e_A, -e_B, np.zeros((m + p, 1))])

    # w y beta libres; y,z >= 0
    bounds = [(0, float("inf"))] * (2 * (m + p))
    sizes = (m, n, p)
    return c, M, b, bounds, sizes


def simplex_estandar(c, M, b, bounds, iters=5000):
    """
    Resuelve un problema lineal estándar con simplex clásico de SciPy.
    Devuelve (x_opt, meta, res) donde 'meta' contiene iteraciones, cpu, etc.
    """
    t0 = time.perf_counter()
    res = linprog(
        c,
        None,
        None,
        M,
        b,
        bounds,
        method="simplex",
        options={"maxiter": iters},
    )
    t1 = time.perf_counter()
    meta = {
        "exito": bool(res.success),
        "msg": res.message,
        "obj": float(res.fun) if res.success else np.inf,
        "iters": getattr(res, "nit", None),
        "tiempo": t1 - t0,
    }
    return res.x, meta, res


# -------------------- KARUSH - KUHN - TUCKER --------------------


def verificar_kkt(
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    x_opt: np.ndarray,
    y_opt: np.ndarray,
) -> dict:
    """
    Condiciones de Karush-Kuhn-Tucker:
      1.- x.T@z ~ 0         (complementariedad)
      2.- A@x-b ~ 0         (factibilidad primal)
      3.- A.T@y+z-c ~ 0     (estacionariedad)
      4.- x, z >= 0         (no negatividad)
    """
    ATy = A.T @ y_opt
    z = c - ATy
    kkt1 = x_opt.T @ z
    kkt2 = float(np.max(A @ x_opt - b))
    kkt3 = float(np.max(ATy + z - c))
    kkt4_1 = np.min(x_opt)
    kkt4_2 = np.min(z)

    return {
        "complementariedad": kkt1,
        "factibilidad_primal": kkt2,
        "estacionariedad": kkt3,
        "no_negatividad_primal": kkt4_1,
        "no_negatividad_holgura_dual": kkt4_2,
    }


# -------------------- gráficas pedidas --------------------


def plot_Aw_plus_y(
    A: np.ndarray, w: np.ndarray, y: np.ndarray, path_png: str
) -> None:
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


def plot_Bw_minus_z(
    B: np.ndarray, w: np.ndarray, z: np.ndarray, path_png: str
) -> None:
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


def plot_pca_hyperplane(X_df: pd.DataFrame, y_labels, w: np.ndarray, beta: float, path_png: str):
    """
    PCA 2D sobre datos ESTANDARIZADOS (z-score) y recta del hiperplano proyectada.
    El PL pudo haberse resuelto en el espacio original; aquí convertimos (w,beta)
    al sistema estandarizado antes de proyectar.
    """

    # 1) Estandarizar columnas (z-score) SOLO para la visualización
    X = X_df.to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    Z = (X - mu) / sigma  # n_samples x n_features

    # 2) PCA en el espacio estandarizado
    pca = PCA(n_components=2, random_state=0)
    Z2 = pca.fit_transform(Z)            # n_samples x 2

    # 3) Convertir hiperplano (w,beta) al sistema estandarizado
    w_std = sigma * w                    # elemento a elemento
    beta_std = beta - float(mu @ w)

    # 4) Proyectar el hiperplano a 2D (PC1, PC2)
    w2 = pca.components_ @ w_std         # 2-vector
    x1 = np.linspace(Z2[:, 0].min(), Z2[:, 0].max(), 400)
    xline = None if abs(w2[1]) < 1e-12 else (beta_std - w2[0] * x1) / w2[1]

    # 5) Dispersión con etiquetas (acepta Series o ndarray)
    y_arr = np.asarray(y_labels)
    maskM = (y_arr == "M")

    plt.figure()
    plt.scatter(Z2[~maskM, 0], Z2[~maskM, 1], s=14, label="B")
    plt.scatter(Z2[ maskM, 0], Z2[ maskM, 1], s=14, label="M")
    if xline is None:
        x0 = beta_std / (w2[0] if abs(w2[0]) > 1e-12 else 1.0)
        plt.axvline(x=x0, linestyle="--", label="Hiperplano (aprox)")
    else:
        plt.plot(x1, xline, "--", label="Hiperplano (aprox)")
    plt.title("PCA 2D con hiperplano proyectado")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()


# -------------------- OBTENCIÓN DEL HIPERPLANO --------------------


def calcular_hiperplano(outdir: str = "outputs_simplex"):
    """
    Ejecuta el problema con SIMPLEX (primal y dual) y regresa un dict con:
      - iteraciones, cpu y objetivo (primal y dual)
      - brecha de dualidad
      - métricas KKT
      - rutas de las figuras Aw+y y Bw-z

    Guarda 'results_simplex.json' y las imágenes en 'outputs_simplex'.
    """
    ensure_dir(outdir)

    # 1) Datos
    X, y_labels = load_breast_cancer()
    A, B = split_A_B(X, y_labels)

    # 2) PRIMAL
    c_prim, M_prim, b_prim, bounds_prim, sizes_prim = construir_primal(A, B)
    opt_prim, meta_prim, res_prim = simplex_estandar(
        c_prim, M_prim, b_prim, bounds_prim
    )

    # Solución del hiperplano
    m, n, p = sizes_prim
    beta = opt_prim[0] - opt_prim[1]
    w = opt_prim[2: 2 + n] - opt_prim[2 + n: 2 + 2 * n]
    y_sl = opt_prim[2 + 2 * n: 2 + 2 * n + m]  # <- antes 'y'
    z_sl = opt_prim[2 + 2 * n + m: 2 + 2 * n + m + p]  # <- antes 'z'

    # 3) DUAL
    c_dual, M_dual, b_dual, bounds_dual, sizes_dual = construir_dual(A, B)
    opt_dual, meta_dual, res_dual = simplex_estandar(
        c_dual, M_dual, b_dual, bounds_dual
    )

    # 4) KKT (usando los lambdas del dual explícito)
    y_opt = np.hstack((opt_dual[:m], -opt_dual[m : m + p]))
    kkt = verificar_kkt(c_prim, M_prim, b_prim, opt_prim, y_opt)

    # 5) Brecha de dualidad
    gap = (
        abs(meta_prim["obj"] - meta_dual["obj"])
        if (meta_prim["exito"] and meta_dual["exito"])
        else None
    )
    # 6) Gráficas pedidas
    fig1 = os.path.join(outdir, "Aw_plus_y_simplex.png")
    fig2 = os.path.join(outdir, "Bw_minus_z_simplex.png")
    fig3 = os.path.join(outdir, "pca_hyperplane_simplex.png")

    plot_Aw_plus_y(A, w, y_sl, fig1)  # usar y_sl (holguras de A)
    plot_Bw_minus_z(B, w, z_sl, fig2)  # usar z_sl (holguras de B)
    plot_pca_hyperplane(X, y_labels, w, beta, fig3)  # pasar etiquetas reales

    # 7) Empaquetado de resultados
    results = (meta_prim, meta_dual, gap, kkt, fig1, fig2, fig3)

    # 8) Guardar JSON
    save_json(os.path.join(outdir, "results_simplex.json"), results)
    return results


# -------------------- Ejecución --------------------

if __name__ == "__main__":
    meta_prim, meta_dual, gap, kkt, fig1, fig2, fig3 = calcular_hiperplano()
    # Impresión de resultados
    print("\n===== SIMPLEX — Resumen =====")
    print("\nEjecución del primal")
    for x, y in meta_prim.items():
        print(x, y)
    print("\nEjecución del dual")
    for x, y in meta_dual.items():
        print(x, y)
    print(f"\nBrecha entre primal y dual = {gap}")
    print("\nSatisfacción de condiciones Karush-Kuhn-Tucker")
    for x, y in kkt.items():
        print(x, y)
    print("Figuras guardadas:")
    print(" -", fig1)
    print(" -", fig2)
    print(" -", fig3)
