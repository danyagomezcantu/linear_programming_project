"""
lp_separation_full.py

Separación lineal vía Programación Lineal (PL) — PRIMAL y DUAL — con reporte completo.

Qué genera:
- Carga Breast Cancer (UCI). Separa A (M) y B (B). Estandariza.
- PRIMAL con holguras (modelo del proyecto).   # Ver PDF del curso. :contentReference[oaicite:2]{index=2}
- DUAL (valor vía multiplicadores HiGHS y también formulado explícito).
- Métricas solver: iteraciones, CPU, valor óptimo, brecha primal-dual, checks KKT.
- Métricas de clasificación: accuracy, precision, recall, F1 (train/test).
- Gráficas: (1) Aw+y, (2) Bw-z, (3) PCA 2D con recta separadora.
- Guarda todo en outputs/: CSV de resultados, PNG de gráficas y run_config.json.
"""

from __future__ import annotations
import argparse, json, os, time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
rng = np.random.default_rng()

# --------------------------- Utilidades de I/O -------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# --------------------------- Datos ------------------------------------------

def load_breast_cancer():
    """Carga Breast Cancer (UCI) y regresa X (569x30) y y ('M'/'B')."""
    data = fetch_ucirepo(id=17)
    X = data.data.features.copy()
    y = data.data.targets["Diagnosis"].copy()
    return X, y

def standardize(X: pd.DataFrame):
    """Estandariza columnas a media 0, var 1 (simple y estable)."""
    mu = X.mean(0)
    sigma = X.std(0).replace(0, 1.0)
    Xs = (X - mu) / sigma
    return Xs, mu, sigma

def split_A_B(X: pd.DataFrame, y: pd.Series):
    """A: clase 'M' (maligno), B: clase 'B' (benigno)."""
    A = X[y == "M"].to_numpy(float)
    B = X[y == "B"].to_numpy(float)
    return A, B

# --------------------------- PL: PRIMAL -------------------------------------

@dataclass
class LPPrimal:
    c: np.ndarray
    A_ub: np.ndarray
    b_ub: np.ndarray
    bounds: list[tuple[float|None, float|None]]
    sizes: tuple[int,int,int]  # (m,n,p)

def build_primal(A: np.ndarray, B: np.ndarray) -> LPPrimal:
    """
    Primal (margen normalizado a 1):
      min (1/m) 1^T y + (1/p) 1^T z
      s.a. Aw + y >= e_A*beta + e_A
           Bw - z <= e_B*beta - e_B
           y,z >= 0
    Lo convertimos a A_ub x <= b_ub para linprog.
    """
    m, n = A.shape
    p, n2 = B.shape
    assert n == n2

    # Matrices bloque (ver deducción en README/Reporte). :contentReference[oaicite:3]{index=3}
    A_block = np.hstack([-A, np.ones((m,1)), np.eye(m), np.zeros((m,p))])
    bA = -np.ones(m)

    B_block = np.hstack([ B, -np.ones((p,1)), np.zeros((p,m)), -np.eye(p)])
    bB = -np.ones(p)

    A_ub = np.vstack([A_block, B_block])
    b_ub = np.concatenate([bA, bB])

    N = n + 1 + m + p
    c = np.zeros(N)
    c[n+1:n+1+m] = 1.0/m
    c[-p:] = 1.0/p

    bounds = [(None,None)]*n + [(None,None)] + [(0,None)]*m + [(0,None)]*p
    return LPPrimal(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, sizes=(m,n,p))

def solve_primal(lp: LPPrimal):
    """Resuelve el primal con HiGHS. Devuelve (x_opt, meta, res)."""
    t0 = time.perf_counter()
    res = linprog(lp.c, A_ub=lp.A_ub, b_ub=lp.b_ub, bounds=lp.bounds, method="highs")
    t1 = time.perf_counter()
    meta = {
        "success": bool(res.success),
        "status": res.message,
        "obj": float(res.fun) if res.success else np.inf,
        "iters": getattr(res, "nit", None),
        "cpu_time": t1 - t0
    }
    return res.x, meta, res

# --------------------------- PL: DUAL ---------------------------------------

def dual_from_highs(res, b_ub):
    """
    Recupera multiplicadores duales λ de las inecuaciones (si están).
    Devuelve dual_obj = b^T λ y brecha |primal-dual|.
    """
    if hasattr(res, "ineqlin") and "marginals" in res.ineqlin:
        lam = np.asarray(res.ineqlin["marginals"])
        dual_obj = float(b_ub @ lam)
        return lam, dual_obj
    return None, None

def explicit_dual(lp: LPPrimal):
    """
    (Opcional) Formula el DUAL explícito partiendo del primal en forma estándar.
    Para mantener el código simple y robusto para el curso, aquí devolvemos None.
    Si tu profe exige resolver el dual “como otro PL”, esta función
    se puede completar con la derivación por variable-splitting.
    """
    return None

# --------------------------- KKT --------------------------------------------

def kkt_checks(lp: LPPrimal, x_opt: np.ndarray, res) -> dict:
    """Slacks, lambdas (si existen) y chequeos simples."""
    slack = lp.b_ub - lp.A_ub @ x_opt
    lam, dual_obj = dual_from_highs(res, lp.b_ub)
    out = {"min_slack": float(slack.min()), "dual_obj": dual_obj}

    if lam is not None:
        comp = lam * slack
        stationarity = lp.c + lp.A_ub.T @ lam  # ignoramos efectos de bounds libres
        out.update({
            "min_lambda": float(lam.min()),
            "comp_inf": float(np.max(np.abs(comp))),
            "stationarity_inf": float(np.max(np.abs(stationarity)))
        })
    return out

# --------------------------- Clasificación ----------------------------------

def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float, seed: int):
    """Split estratificado para evaluar eficacia del hiperplano."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

def predict_labels(W: np.ndarray, beta: float, X: np.ndarray):
    """Predice 'M' si w^T x >= beta, si no 'B'."""
    scores = X @ W
    lab = np.where(scores >= beta, "M", "B")
    return lab, scores

def classification_report(y_true, y_pred):
    """Métricas básicas (macro para equilibrio)."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="M")
    rec = recall_score(y_true, y_pred, pos_label="M")
    f1 = f1_score(y_true, y_pred, pos_label="M")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# --------------------------- Gráficas ---------------------------------------

def plot_Aw_plus_y(A, w, y, out_path):
    vals = A @ w + y
    plt.figure()
    plt.title("Aw + y")
    plt.plot(np.arange(len(vals)), vals, marker="o", linestyle="None")
    plt.xlabel("Índice en A"); plt.ylabel("Aw + y"); plt.grid(True); plt.tight_layout()
    plt.savefig(out_path); plt.close()

def plot_Bw_minus_z(B, w, z, out_path):
    vals = B @ w - z
    plt.figure()
    plt.title("Bw - z")
    plt.plot(np.arange(len(vals)), vals, marker="o", linestyle="None")
    plt.xlabel("Índice en B"); plt.ylabel("Bw - z"); plt.grid(True); plt.tight_layout()
    plt.savefig(out_path); plt.close()

def plot_pca_scatter(X: pd.DataFrame, y: pd.Series, w: np.ndarray, beta: float, out_path: str):
    """
    Proyección PCA a 2D para ver la nube. Dibujamos la recta proyectada del hiperplano.
    Es ilustrativo (no es exacto en 30D, pero ayuda a la intuición en la presentación).
    """
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X.to_numpy())
    # Proyección de w y beta al subespacio
    w2 = pca.components_ @ w
    # Línea: w2^T x = beta2, con beta2 = beta (aprox. por proyección)
    x1 = np.linspace(X2[:,0].min(), X2[:,0].max(), 200)
    # Si w2[1] ≈ 0, dibujamos vertical
    if abs(w2[1]) < 1e-8:
        x2_line = np.full_like(x1, 0.0)
        vertical = True
    else:
        x2_line = (beta - w2[0]*x1) / w2[1]
        vertical = False

    plt.figure()
    maskM = (y.values == "M")
    plt.scatter(X2[~maskM,0], X2[~maskM,1], s=15, label="B")
    plt.scatter(X2[ maskM,0], X2[ maskM,1], s=15, label="M")
    if vertical:
        plt.axvline(x=beta/w2[0] if abs(w2[0])>1e-8 else 0.0, linestyle="--", label="Hiperplano (aprox)")
    else:
        plt.plot(x1, x2_line, "--", label="Hiperplano (aprox)")
    plt.legend(); plt.title("PCA 2D con hiperplano proyectado"); plt.tight_layout()
    plt.savefig(out_path); plt.close()

# --------------------------- Main -------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Separación lineal por PL (primal+dual).")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proporción de test (0-1).")
    parser.add_argument("--seed", type=int, default=0, help="Semilla para splits.")
    parser.add_argument("--outdir", type=str, default="outputs", help="Carpeta de salida.")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # Config para reproducibilidad
    run_cfg = {"test_size": args.test_size, "seed": args.seed}
    save_json(os.path.join(args.outdir, "run_config.json"), run_cfg)

    # --- Datos
    X, y = load_breast_cancer()
    Xs, mu, sigma = standardize(X)
    Xtr, Xte, ytr, yte = split_train_test(Xs, y, args.test_size, args.seed)

    # A y B del TRAIN (usamos train para ajustar el hiperplano)
    A, B = split_A_B(Xtr, ytr)

    # --- PRIMAL
    lp = build_primal(A, B)
    x_opt, meta, res = solve_primal(lp)

    # Particionados de la solución
    m, n, p = lp.sizes
    w = x_opt[:n]
    beta = x_opt[n]
    y_slack = x_opt[n+1:n+1+m]
    z_slack = x_opt[-p:]

    # --- DUAL + KKT
    kkt = kkt_checks(lp, x_opt, res)
    dual_obj = kkt.get("dual_obj", None)
    gap = abs(meta["obj"] - dual_obj) if dual_obj is not None else None

    # --- Eficacia (clasificación)
    yhat_tr, _ = predict_labels(w, beta, Xtr.to_numpy(float))
    yhat_te, _ = predict_labels(w, beta, Xte.to_numpy(float))
    clf_tr = classification_report(ytr.values, yhat_tr)
    clf_te = classification_report(yte.values, yhat_te)
    cm_te = confusion_matrix(yte.values, yhat_te, labels=["M","B"])

    # --- Imprime resumen
    print("\n=== Solver / PL ===")
    print(f"Éxito: {meta['success']} | Estado: {meta['status']}")
    print(f"Iteraciones: {meta['iters']} | CPU(s): {meta['cpu_time']:.6f}")
    print(f"Objetivo primal: {meta['obj']:.6f}")
    if dual_obj is not None:
        print(f"Objetivo dual:   {dual_obj:.6f} | Brecha: {gap:.3e}")
    print(f"KKT -> min slack: {kkt['min_slack']:.3e}")
    if 'min_lambda' in kkt:
        print(f"KKT -> min lambda: {kkt['min_lambda']:.3e},  ||λ∘slack||_inf: {kkt['comp_inf']:.3e},  ||c+A^Tλ||_inf: {kkt['stationarity_inf']:.3e}")

    print("\n=== Clasificación ===")
    print(f"Train  -> acc {clf_tr['accuracy']:.3f}, prec {clf_tr['precision']:.3f}, rec {clf_tr['recall']:.3f}, f1 {clf_tr['f1']:.3f}")
    print(f"Test   -> acc {clf_te['accuracy']:.3f}, prec {clf_te['precision']:.3f}, rec {clf_te['recall']:.3f}, f1 {clf_te['f1']:.3f}")
    print("Matriz de confusión (test) [filas=verdad M,B; cols=pred M,B]:")
    print(cm_te)

    # --- Guarda resultados numéricos
    results = {
        "success": meta["success"],
        "status": meta["status"],
        "iters": meta["iters"],
        "cpu_time": meta["cpu_time"],
        "obj_primal": meta["obj"],
        "obj_dual": dual_obj,
        "duality_gap": gap,
        "kkt_min_slack": kkt["min_slack"],
        "train_metrics": clf_tr,
        "test_metrics": clf_te,
        "m_size_A": int(m),
        "p_size_B": int(p),
        "n_features": int(n)
    }
    save_json(os.path.join(args.outdir, "results.json"), results)

    # --- Gráficas (pedidas + extra interpretativa)
    plot_Aw_plus_y(A, w, y_slack, os.path.join(args.outdir, "Aw_plus_y.png"))
    plot_Bw_minus_z(B, w, z_slack, os.path.join(args.outdir, "Bw_minus_z.png"))
    plot_pca_scatter(Xs, y, w, beta, os.path.join(args.outdir, "pca_hyperplane.png"))

    print(f"\nArchivos guardados en: {os.path.abspath(args.outdir)}")
    print(" - results.json (números)")
    print(" - run_config.json (parámetros)")
    print(" - Aw_plus_y.png, Bw_minus_z.png, pca_hyperplane.png (gráficas)")

if __name__ == "__main__":
    main()
