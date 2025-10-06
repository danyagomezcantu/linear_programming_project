"""
Compara resultados entre:
  A) Simplex  (primal + dual explícito)          
  B) HiGHS    (primal; dual por multiplicadores)  # b^T λ y KKT con λ de HiGHS

  - iteraciones, CPU (s)
  - objetivo PRIMAL
  - objetivo DUAL (max-form)
  - brecha |primal - dual|
  - KKT: min_slack, ||λ∘slack||_inf, ||c + A^T λ||_inf
  - Gráficas: Aw+y, Bw−z, PCA 2D (con sufijo _simplex o _highs)

Requisitos:
  pip install numpy pandas scipy matplotlib scikit-learn ucimlrepo
"""

from __future__ import annotations
import os, time, json
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import linprog
from ucimlrepo import fetch_ucirepo


# ========================= utilidades I/O =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ========================= datos =========================

def load_breast_cancer() -> Tuple[pd.DataFrame, pd.Series]:
    data = fetch_ucirepo(id=17)
    X = data.data.features.copy()
    y = data.data.targets["Diagnosis"].copy()  # 'M'/'B'
    return X, y

def standardize(X: pd.DataFrame) -> pd.DataFrame:
    mu = X.mean(0)
    sigma = X.std(0).replace(0, 1.0)
    return (X - mu) / sigma

def split_A_B(Xs: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    A = Xs[y == "M"].to_numpy(float)
    B = Xs[y == "B"].to_numpy(float)
    return A, B


# ========================= gráficas comunes =========================

def plot_Aw_plus_y(A, w, y, path_png):
    vals = A @ w + y
    plt.figure()
    plt.title("Aw + y"); plt.plot(np.arange(len(vals)), vals, marker="o", linestyle="None")
    plt.xlabel("Índice en A"); plt.ylabel("Aw + y"); plt.grid(True); plt.tight_layout()
    plt.savefig(path_png); plt.close()

def plot_Bw_minus_z(B, w, z, path_png):
    vals = B @ w - z
    plt.figure()
    plt.title("Bw - z"); plt.plot(np.arange(len(vals)), vals, marker="o", linestyle="None")
    plt.xlabel("Índice en B"); plt.ylabel("Bw - z"); plt.grid(True); plt.tight_layout()
    plt.savefig(path_png); plt.close()

def plot_pca_hyperplane(Xs: pd.DataFrame, y: pd.Series, w: np.ndarray, beta: float, path_png: str):
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(Xs.to_numpy())
    w2 = pca.components_ @ w
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
        x0 = beta / (w2[0] if abs(w2[0])>1e-12 else 1.0)
        plt.axvline(x=x0, linestyle="--", label="Hiperplano (aprox)")
    else:
        plt.plot(x1, xline, "--", label="Hiperplano (aprox)")
    plt.title("PCA 2D con hiperplano proyectado")
    plt.legend(); plt.tight_layout(); plt.savefig(path_png); plt.close()


# ========================= PRIMAL/DUAL — SIMPLEX =========================
# (idéntico enfoque a tu script corregido)  :contentReference[oaicite:3]{index=3}

def build_primal_eq(A: np.ndarray, B: np.ndarray):
    m, n = A.shape
    p = B.shape[0]
    M_A = np.hstack([-np.ones((m,1)), +np.ones((m,1)), +A, -A, np.eye(m), np.zeros((m,p)), -np.eye(m), np.zeros((m,p))])
    b_A = np.ones(m)
    M_B = np.hstack([-np.ones((p,1)), +np.ones((p,1)), +B, -B, np.zeros((p,m)), -np.eye(p), np.zeros((p,m)), +np.eye(p)])
    b_B = -np.ones(p)
    M = np.vstack([M_A, M_B]); b = np.concatenate([b_A, b_B])
    N = 2 + 2*n + m + p + m + p
    c = np.zeros(N)
    c[2+2*n:2+2*n+m] = 1.0/m
    c[2+2*n+m:2+2*n+m+p] = 1.0/p
    bounds = [(0, None)]*N
    return c, M, b, bounds, (m,n,p)

def build_dual_eq(A: np.ndarray, B: np.ndarray):
    m, n = A.shape; p = B.shape[0]
    eA = np.ones((m,1)); eB = np.ones((p,1))
    M1 = np.hstack([ eA.T, -eB.T, np.zeros((1,m+p)) ])
    M2 = np.hstack([ A.T,  -B.T,  np.zeros((n,m+p)) ])
    M3 = np.hstack([ np.eye(m), np.zeros((m,p)),  np.eye(m), np.zeros((m,p)) ])
    M4 = np.hstack([ np.zeros((p,m)), np.eye(p), np.zeros((p,m)), np.eye(p) ])
    M = np.vstack([M1, M2, M3, M4])
    b = np.concatenate([ np.zeros(1+n), (eA/m).ravel(), (eB/p).ravel() ])
    c = np.concatenate([ -np.ones(m), -np.ones(p), np.zeros(m+p) ])  # min -sum(u)-sum(q)
    bounds = [(0,None)]*(m+p+m+p)
    return c, M, b, bounds

def build_primal_ub(A: np.ndarray, B: np.ndarray):
    m, n = A.shape; p = B.shape[0]
    A_block = np.hstack([ -A, np.ones((m,1)), np.eye(m), np.zeros((m,p)) ])
    B_block = np.hstack([  B, -np.ones((p,1)), np.zeros((p,m)), -np.eye(p) ])
    A_ub = np.vstack([A_block, B_block])
    b_ub = -np.ones(m+p)
    c = np.zeros(n+1+m+p)
    c[n+1:n+1+m] = 1.0/m
    c[n+1+m:]     = 1.0/p
    bounds = [(None,None)]*n + [(None,None)] + [(0,None)]*m + [(0,None)]*p
    return c, A_ub, b_ub, bounds

def simplex_eq(c, M, b, bounds, maxiter=10000):
    t0 = time.perf_counter()
    res = linprog(c, A_eq=M, b_eq=b, bounds=bounds, method="simplex", options={"maxiter": maxiter})
    t1 = time.perf_counter()
    return res, t1 - t0

def run_simplex_path(Xs: pd.DataFrame, y: pd.Series, outdir: str) -> Dict[str, Any]:
    A, B = split_A_B(Xs, y)
    cP, MP, bP, bndP, (m,n,p) = build_primal_eq(A,B)
    resP, cpuP = simplex_eq(cP, MP, bP, bndP)
    xP = resP.x
    beta = xP[0]-xP[1]; w = xP[2:2+n]-xP[2+n:2+2*n]
    y_sl = xP[2+2*n:2+2*n+m]; z_sl = xP[2+2*n+m:2+2*n+m+p]

    cD, MD, bD, bndD = build_dual_eq(A,B)
    resD, cpuD = simplex_eq(cD, MD, bD, bndD)
    xD = resD.x; u = xD[:m]; q = xD[m:m+p]
    obj_dual_max = -float(resD.fun)

    # KKT (con inecuaciones originales)
    c_v, A_ub, b_ub, _ = build_primal_ub(A,B)
    v = np.concatenate([w,[beta],y_sl,z_sl]); lam = np.concatenate([u,q])
    slack = b_ub - A_ub @ v
    kkt = {
        "min_slack": float(slack.min()),
        "comp_inf": float(np.max(np.abs(lam*slack))),
        "station_inf": float(np.max(np.abs(c_v + A_ub.T @ lam))),
    }

    # Gráficas
    fig1 = os.path.join(outdir, "Aw_plus_y_simplex.png")
    fig2 = os.path.join(outdir, "Bw_minus_z_simplex.png")
    fig3 = os.path.join(outdir, "pca_hyperplane_simplex.png")
    plot_Aw_plus_y(A, w, y_sl, fig1)
    plot_Bw_minus_z(B, w, z_sl, fig2)
    plot_pca_hyperplane(Xs, y, w, beta, fig3)

    return {
        "sizes": {"m": int(m), "p": int(p), "n": int(n)},
        "primal": {"success": bool(resP.success), "status": resP.message, "iterations": getattr(resP, "nit", None),
                   "cpu_seconds": cpuP, "objective": float(resP.fun)},
        "dual":   {"success": bool(resD.success), "status": resD.message, "iterations": getattr(resD, "nit", None),
                   "cpu_seconds": cpuD, "objective_max_form": obj_dual_max},
        "gap": abs(float(resP.fun) - obj_dual_max) if (resP.success and resD.success) else None,
        "KKT": kkt,
        "figures": {"Aw_plus_y": fig1, "Bw_minus_z": fig2, "pca_2d": fig3},
        "w_beta": {"w_norm": float(np.linalg.norm(w)), "beta": float(beta)}
    }


# ========================= PRIMAL/DUAL — HiGHS =========================
# (formulación en <=; dual y KKT por λ de HiGHS)  :contentReference[oaicite:4]{index=4}

def highs_primal(A: np.ndarray, B: np.ndarray):
    c, A_ub, b_ub, bounds = build_primal_ub(A,B)
    t0 = time.perf_counter()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    t1 = time.perf_counter()
    cpu = t1 - t0
    x = res.x
    n = A.shape[1]; m = A.shape[0]; p = B.shape[0]
    w = x[:n]; beta = x[n]; y = x[n+1:n+1+m]; z = x[n+1+m:]
    # Dual por multiplicadores (si disponibles)
    dual_obj, lam = None, None
    if hasattr(res, "ineqlin") and isinstance(res.ineqlin, dict) and "marginals" in res.ineqlin:
        lam = np.asarray(res.ineqlin["marginals"])
        dual_obj = float(b_ub @ lam)
    # KKT
    slack = b_ub - A_ub @ x
    kkt = {
        "min_slack": float(slack.min()),
        "comp_inf": float(np.max(np.abs((lam if lam is not None else np.zeros_like(slack)) * slack))),
        "station_inf": float(np.max(np.abs(c + A_ub.T @ (lam if lam is not None else np.zeros_like(slack))))),
    }
    meta = {"success": bool(res.success), "status": res.message, "iterations": getattr(res, "nit", None),
            "cpu_seconds": cpu, "objective": float(res.fun)}
    return w, beta, y, z, meta, dual_obj, (c, A_ub, b_ub)

def run_highs_path(Xs: pd.DataFrame, y: pd.Series, outdir: str) -> Dict[str, Any]:
    A, B = split_A_B(Xs, y)
    w, beta, y_sl, z_sl, metaP, dual_obj, pack = highs_primal(A,B)
    c, A_ub, b_ub = pack
    # Gráficas
    fig1 = os.path.join(outdir, "Aw_plus_y_highs.png")
    fig2 = os.path.join(outdir, "Bw_minus_z_highs.png")
    fig3 = os.path.join(outdir, "pca_hyperplane_highs.png")
    plot_Aw_plus_y(A, w, y_sl, fig1)
    plot_Bw_minus_z(B, w, z_sl, fig2)
    plot_pca_hyperplane(Xs, y, w, beta, fig3)

    return {
        "sizes": {"m": int(A.shape[0]), "p": int(B.shape[0]), "n": int(A.shape[1])},
        "primal": metaP,
        "dual": {"objective_max_form": float(dual_obj) if dual_obj is not None else None,
                 "note": "dual via HiGHS marginals (b^T λ)"},
        "gap": abs(metaP["objective"] - dual_obj) if (metaP["success"] and dual_obj is not None) else None,
        "KKT": {
            "min_slack": float((b_ub - A_ub @ np.concatenate([w,[beta],y_sl,z_sl])).min()),
            "comp_inf": float(np.max(np.abs(( (getattr(linprog, '__dummy__', None)) ))))  # placeholder overwritten below
        },
        "figures": {"Aw_plus_y": fig1, "Bw_minus_z": fig2, "pca_2d": fig3},
        "w_beta": {"w_norm": float(np.linalg.norm(w)), "beta": float(beta)}
    }

# sobrescribimos KKT correctamente (evitamos dependencia al truco anterior)
def _kkt_highs(A,B,w,beta,y_sl,z_sl,c,A_ub,b_ub,lam):
    v = np.concatenate([w,[beta],y_sl,z_sl])
    slack = b_ub - A_ub @ v
    comp_inf = float(np.max(np.abs((lam if lam is not None else np.zeros_like(slack)) * slack)))
    station_inf = float(np.max(np.abs(c + A_ub.T @ (lam if lam is not None else np.zeros_like(slack)))))
    return float(slack.min()), comp_inf, station_inf


# ========================= comparación maestro =========================

def run_compare(outdir="outputs_compare") -> Dict[str, Any]:
    ensure_dir(outdir)
    # Datos
    X, y = load_breast_cancer()
    Xs = standardize(X)

    # SIMPLEX
    outS = run_simplex_path(Xs, y, outdir)
    # HiGHS
    A, B = split_A_B(Xs, y)
    c, A_ub, b_ub, _ = build_primal_ub(A,B)
    wH, betaH, yH, zH, metaH, dualH, _pack = highs_primal(A,B)
    # KKT de HiGHS bien formadas
    lamH = None
    # Para obtener lamH de nuevo:
    res_tmp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None,None)]*A_ub.shape[1], method="highs")
    if hasattr(res_tmp, "ineqlin") and "marginals" in res_tmp.ineqlin:
        lamH = np.asarray(res_tmp.ineqlin["marginals"])
    min_slackH, comp_infH, station_infH = _kkt_highs(A,B,wH,betaH,yH,zH,c,A_ub,b_ub,lamH)

    # Gráficas HiGHS
    fig1H = os.path.join(outdir, "Aw_plus_y_highs.png")
    fig2H = os.path.join(outdir, "Bw_minus_z_highs.png")
    fig3H = os.path.join(outdir, "pca_hyperplane_highs.png")
    plot_Aw_plus_y(A, wH, yH, fig1H)
    plot_Bw_minus_z(B, wH, zH, fig2H)
    plot_pca_hyperplane(Xs, y, wH, betaH, fig3H)

    outH = {
        "sizes": {"m": int(A.shape[0]), "p": int(B.shape[0]), "n": int(A.shape[1])},
        "primal": metaH,
        "dual": {"objective_max_form": float(dualH) if dualH is not None else None,
                 "note": "dual via HiGHS marginals (b^T λ)"},
        "gap": abs(metaH["objective"] - dualH) if (metaH["success"] and dualH is not None) else None,
        "KKT": {"min_slack": min_slackH, "comp_inf": comp_infH, "station_inf": station_infH},
        "figures": {"Aw_plus_y": fig1H, "Bw_minus_z": fig2H, "pca_2d": fig3H},
        "w_beta": {"w_norm": float(np.linalg.norm(wH)), "beta": float(betaH)}
    }

    # Comparación consolidada
    results = {"simplex": outS, "highs": outH}
    save_json(os.path.join(outdir, "results_compare.json"), results)
    return results


# ========================= CLI =========================

if __name__ == "__main__":
    out = run_compare()
    S, H = out["simplex"], out["highs"]
    print("\n=== COMPARACIÓN SIMPLEX vs HiGHS ===")
    print(f"Datos: n={S['sizes']['n']}, m={S['sizes']['m']} (A), p={S['sizes']['p']} (B)")
    print("\n-- SIMPLEX --")
    print(f"PRIMAL -> iters={S['primal']['iterations']}, cpu={S['primal']['cpu_seconds']:.6f}s, obj={S['primal']['objective']:.6f}")
    print(f"DUAL   -> iters={S['dual']['iterations']},  cpu={S['dual']['cpu_seconds']:.6f}s, obj(max)={S['dual']['objective_max_form']:.6f}")
    print(f"Gap |P-D| = {S['gap']}")
    print(f"KKT  -> min_slack={S['KKT']['min_slack']:.3e}, ||λ∘s||_inf={S['KKT']['comp_inf']:.3e}, ||c+A^Tλ||_inf={S['KKT']['station_inf']:.3e}")
    print("Figuras:", S["figures"])

    print("\n-- HiGHS --")
    print(f"PRIMAL -> iters={H['primal']['iterations']}, cpu={H['primal']['cpu_seconds']:.6f}s, obj={H['primal']['objective']:.6f}")
    print(f"DUAL   -> obj(max)={H['dual']['objective_max_form']}")
    print(f"Gap |P-D| = {H['gap']}")
    print(f"KKT  -> min_slack={H['KKT']['min_slack']:.3e}, ||λ∘s||_inf={H['KKT']['comp_inf']:.3e}, ||c+A^Tλ||_inf={H['KKT']['station_inf']:.3e}")
    print("Figuras:", H["figures"])
