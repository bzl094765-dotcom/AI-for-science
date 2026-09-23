#!/usr/bin/env python3
"""Minimal CISF HSIC repro on the official emotions.mat.

Faithful NumPy port of train_L / Update_C / Update_WC / Update_WD / Update_S
from the author zip (gengyulyu.github.io homepage assets/codes/CISF.zip).
Not a paper-table reproduction: run.m never assigns `para`, and MATLAB/Octave
are not installed. Diagnostic parameter values are labeled in the log.
"""
from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import scipy.io as sio

MAT = Path("/tmp/author_code/CISF/dataset/emotions.mat")
EPS = np.finfo(np.float64).eps
Q = 6
V = 2


def l2_row_then_bias(X_nd: np.ndarray) -> np.ndarray:
    """normalization.m, type='l2', addEPS=1. Rows are samples."""
    data = X_nd + EPS
    denom = np.sqrt(np.sum(data ** 2, axis=1, keepdims=True))
    normed = data / denom
    ones = np.ones((normed.shape[0], 1), dtype=np.float64)
    return np.concatenate([normed, ones], axis=1)


def load_emotions_as_run_m():
    raw = sio.loadmat(MAT)
    # data is 1 x V cell, each d x n; target is q x n. Mirror run.m transposes.
    cells = [np.array(raw["data"][0, i], dtype=np.float64) for i in range(V)]
    Y = np.array(raw["target"], dtype=np.float64)
    X = []
    for view in cells:
        nd = view.T  # n x d
        nd = l2_row_then_bias(nd)
        X.append(nd.T.copy())  # (d+1) x n
    return X, Y


def frob(A):
    return float(np.linalg.norm(A, "fro"))


def safe_stats(name, A):
    A = np.asarray(A, dtype=np.float64)
    finite = np.isfinite(A)
    out = {
        "name": name,
        "shape": list(A.shape),
        "finite_frac": float(finite.mean()) if A.size else 1.0,
        "nan": int(np.isnan(A).sum()),
        "posinf": int(np.isposinf(A).sum()),
        "neginf": int(np.isneginf(A).sum()),
    }
    if finite.any():
        vals = A[finite]
        out["min"] = float(vals.min())
        out["max"] = float(vals.max())
        out["fro"] = float(np.linalg.norm(vals))
    else:
        out["min"] = out["max"] = out["fro"] = float("nan")
    return out


def inv_report(P):
    P = np.asarray(P, dtype=np.float64)
    # symmetric part for eig; the code inverts P as written (not symmetrized)
    try:
        eig = np.linalg.eigvalsh(0.5 * (P + P.T))
        emin = float(eig.min())
        emax = float(eig.max())
    except np.linalg.LinAlgError:
        emin = emax = float("nan")
    try:
        s = np.linalg.svd(P, compute_uv=False)
        smax = float(s[0])
        smin = float(s[-1])
        cond = float(smax / smin) if smin > 0 else float("inf")
    except np.linalg.LinAlgError:
        smin = smax = cond = float("nan")
    info = {"emin_sym": emin, "emax_sym": emax, "smin": smin, "smax": smax, "cond": cond}
    try:
        invP = np.linalg.inv(P)
        info["inv_ok"] = True
        info["inv_fro"] = frob(invP)
        info["error"] = ""
        return invP, info
    except np.linalg.LinAlgError as e:
        info["inv_ok"] = False
        info["inv_fro"] = float("nan")
        info["error"] = str(e)
        return None, info


def hsic_linear(Ci, Cj):
    """HSIC with linear kernel and centering, up to the constant (n-1)^{-2}.
    K = C^T C matches the code's C{j}'*C{j} (C is q x n).
    """
    n = Ci.shape[1]
    H = np.eye(n) - np.ones((n, n)) / n
    Ki = Ci.T @ Ci
    Kj = Cj.T @ Cj
    return float(np.trace(H @ Ki @ H @ Kj))


def update_C(X, W_C, S, M, para, C_prev):
    """Update_C.m, including the leaked loop index j."""
    v_num = len(X)
    ins_num = X[0].shape[1]
    H = np.eye(ins_num) - np.ones((ins_num, ins_num)) / ins_num
    C = [S @ W_C[i] @ X[i] for i in range(v_num)]
    reports = []
    j_last = None
    for i in range(v_num):
        K = np.zeros((ins_num, ins_num))
        for j in range(v_num):
            j_last = j
            if j == i:
                continue
            K = K + H @ C[j].T @ C[j] @ H
        # code uses para.mu(j) after the inner loop, not the paired view
        mu_j = para["mu"][j_last]
        P = (
            para["gamma"] * para["mu"][i] * mu_j * (K.T + K)
            - para["lambda"][i] * np.eye(ins_num)
        )
        rhs = M[i] - para["lambda"][i] * (S @ W_C[i] @ X[i])
        invP, info = inv_report(P)
        info["view"] = i
        info["j_last_0based"] = j_last
        info["mu_i"] = float(para["mu"][i])
        info["mu_j_used"] = float(mu_j)
        info["lambda_i"] = float(para["lambda"][i])
        if invP is None:
            C[i] = np.full_like(rhs, np.nan)
            info["C_fro"] = float("nan")
        else:
            C[i] = rhs @ invP
            info["C_fro"] = frob(C[i])
        reports.append(info)
    return C, reports


def update_WC(X, Y, C, M, W_D, S, para):
    v_num = len(X)
    W_C = []
    for i in range(v_num):
        zeta = 0.5 * (2 * para["mu"][i] + para["lambda"][i])
        # code: 1/(2*mu+lambda) * (2*mu*(Y-S*Wd*X) + lambda*C + M)
        temp = (1.0 / (2 * para["mu"][i] + para["lambda"][i])) * (
            2 * para["mu"][i] * (Y - S @ W_D[i] @ X[i])
            + para["lambda"][i] * C[i]
            + M[i]
        )
        U, A, Vt = np.linalg.svd(temp, full_matrices=False)
        sp = A.copy()
        thresh = para["alpha"] / (para["lambda"][i] + 2 * para["mu"][i])
        svp = int(np.sum(sp > thresh))
        if svp >= 1:
            sp = sp[:svp] - thresh
        else:
            svp = 1
            sp = np.array([0.0])
        mid = U[:, :svp] @ np.diag(sp) @ Vt[:svp, :]
        W_C.append(np.linalg.pinv(S) @ mid @ np.linalg.pinv(X[i]))
    return W_C


def update_WD(X, Y, W_C, W_D, S, para, inner=30):
    v_num = len(X)
    W_D = [w.copy() for w in W_D]
    for i in range(v_num):
        temp = np.zeros((Y.shape[0], X[i].shape[0]))
        j_last = None
        for j in range(v_num):
            j_last = j
            if i == j:
                continue
            temp = temp + (
                para["eta"] * para["mu"][i] * para["mu"][j_last] * W_D[j] @ X[j] @ X[i].T
            )
        for _ in range(inner):
            numer = (
                2 * para["mu"][i] * S.T @ Y @ X[i].T
                - 2 * para["mu"][i] * S.T @ S @ W_C[i] @ X[i] @ X[i].T
                - temp
            )
            denom = (
                2 * para["beta"] * W_D[i]
                + 2 * para["mu"][i] * S.T @ S @ W_D[i] @ X[i] @ X[i].T
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                W_D[i] = W_D[i] * (numer / denom)
    return W_D


def update_S(X, Y, W_C, W_D, C, M, para):
    """Update_S.m overwrites S for every view; only the last view survives."""
    v_num = len(X)
    lab_num = Y.shape[0]
    S = np.zeros((lab_num, lab_num))
    for i in range(v_num):
        T0 = W_C[i] + W_D[i]
        S_A = (
            2 * para["mu"][i] * Y @ X[i].T @ T0.T
            + para["lambda"][i] * C[i] @ X[i].T @ W_C[i].T
            + M[i] @ X[i].T @ W_C[i].T
        )
        S_B = (
            2 * para["mu"][i] * T0 @ X[i] @ X[i].T @ T0.T
            + para["lambda"][i] * W_C[i] @ X[i] @ X[i].T @ W_C[i].T
        )
        S = S_A @ np.linalg.inv(S_B + np.eye(S_B.shape[0]))
        S[S < 0] = 0
        s_norms = np.sum(np.abs(S), axis=0)
        norms = np.maximum(s_norms, 1e-30)
        S = np.abs(S) / norms
        S = S - np.diag(np.diag(S)) + np.eye(lab_num)
    return S


def code_style_init(X, Y):
    C, M, W_D, W_C = [], [], [], []
    for i in range(len(X)):
        C.append(Y.copy())
        M.append(Y.copy())
        Wp = Y @ np.linalg.pinv(X[i])
        W_D.append(Wp.copy())
        W_C.append(Wp.copy())
    S = np.eye(Y.shape[0])
    return C, M, W_D, W_C, S


def zero_init(X, Y):
    C, M, W_D, W_C = [], [], [], []
    for i in range(len(X)):
        q, n = Y.shape
        d = X[i].shape[0]
        C.append(np.zeros((q, n)))
        M.append(np.zeros((q, n)))
        W_D.append(np.zeros((q, d)))
        W_C.append(np.zeros((q, d)))
    S = np.eye(Y.shape[0])
    return C, M, W_D, W_C, S


def mu_from_code_formula(X, Y, W_C, W_D, S):
    """train_L.m line 45, applied as an explicit initial value (run.m never does this)."""
    mus = []
    for i in range(len(X)):
        resid = frob(Y - (W_C[i] + S @ W_D[i]) @ X[i])
        mus.append(1.0 / (2.0 * resid))
    return mus


def theorem_lambda(alpha_paper, mu):
    """Theorem 1: lambda >= 8*q*(V-1)*alpha*mu_i*mu_j.
    For V=2 the inner loop always leaves j = view 2, so both views use mu[-1].
    """
    lams = []
    for i in range(V):
        lams.append(8 * Q * (V - 1) * alpha_paper * mu[i] * mu[-1])
    return lams


def one_c_step(tag, X, Y, init, para):
    print(f"\n===== ONE Update_C :: {tag} =====")
    C, M, W_D, W_C, S = init(X, Y)
    print("init", {k: safe_stats(k, v[0] if isinstance(v, list) else v) for k, v in {
        "C0": C, "M0": M, "WD0": W_D, "WC0": W_C, "S": S
    }.items()})
    # HSIC of the matrices actually fed to the kernel (after the overwrite C=S*W*X)
    C_lin = [S @ W_C[i] @ X[i] for i in range(V)]
    print(f"HSIC_before_solve views (0,1) on S*W*X = {hsic_linear(C_lin[0], C_lin[1]):.6e}")
    try:
        C_new, reports = update_C(X, W_C, S, M, para, C)
    except Exception:
        print("Update_C raised:")
        traceback.print_exc()
        return
    for r in reports:
        print("view_report", r)
    if all(np.isfinite(C_new[i]).all() for i in range(V)):
        print(f"HSIC_after views (0,1) = {hsic_linear(C_new[0], C_new[1]):.6e}")
        print("C0_fro", frob(C_new[0]), "C1_fro", frob(C_new[1]))
    else:
        print("C_after not finite")


def short_loop(tag, X, Y, init, para, n_iter=5, tau=1.0):
    print(f"\n===== SHORT LOOP :: {tag}  tau={tau} (tau is NOT in the released code or the extracted paper text) =====")
    C, M, W_D, W_C, S = init(X, Y)
    para = {k: (v.copy() if isinstance(v, list) else v) for k, v in para.items()}
    last_err = 0.0
    for it in range(1, n_iter + 1):
        try:
            W_C = update_WC(X, Y, C, M, W_D, S, para)
            W_D = update_WD(X, Y, W_C, W_D, S, para, inner=30)
            C, reports = update_C(X, W_C, S, M, para, C)
            S = update_S(X, Y, W_C, W_D, C, M, para)
        except Exception as e:
            print(f"iter {it} raised {type(e).__name__}: {e}")
            traceback.print_exc()
            return
        for i in range(V):
            M[i] = M[i] + para["lambda"][i] * (C[i] - S @ W_C[i] @ X[i])
            para["lambda"][i] = min(para["lambda_max"], tau * para["lambda"][i])
            resid_i = frob(Y - (W_C[i] + S @ W_D[i]) @ X[i])
            para["mu"][i] = 1.0 / (2.0 * resid_i) if resid_i > 0 else float("inf")
        # train_L monitors only the last view index left in the loop
        curr_err = frob(Y - (W_C[-1] + S @ W_D[-1]) @ X[-1])
        rel = abs(curr_err - last_err) / curr_err if curr_err > 0 else float("inf")
        c_fin = all(np.isfinite(C[i]).all() for i in range(V))
        w_fin = all(np.isfinite(W_D[i]).all() and np.isfinite(W_C[i]).all() for i in range(V))
        print(
            f"iter {it} curr_err_last_view={curr_err:.6e} rel={rel:.6e} "
            f"C_finite={c_fin} W_finite={w_fin} "
            f"C_fro={[frob(C[i]) if np.isfinite(C[i]).all() else float('nan') for i in range(V)]} "
            f"mu={para['mu']} lambda={para['lambda']} "
            f"inv_ok={[r['inv_ok'] for r in reports]} cond={[r['cond'] for r in reports]}"
        )
        if (not c_fin) or (not w_fin) or curr_err > 1e12:
            print("stop: non-finite or curr_err exploded")
            return
        if rel < 1e-3 and it > 1:
            print("stop: relative residual criterion copied from train_L.m")
            return
        last_err = curr_err


def main():
    X, Y = load_emotions_as_run_m()
    print("MAT", MAT)
    print("views", [x.shape for x in X], "Y", Y.shape, "Y unique", np.unique(Y).tolist())
    print("NOTE run.m references `para` with no assignment. Required fields used by the updates:")
    print("  mu, lambda, gamma (HSIC coeff = paper alpha), alpha (SVT = paper gamma),")
    print("  beta (multiplicative denom = paper eta in this reading), eta (cross term = paper beta),")
    print("  lambda_max, tau, thr (thr is only used at test time in run.m).")

    C, M, WD, WC, S = code_style_init(X, Y)
    mu_code = mu_from_code_formula(X, Y, WC, WD, S)
    print("mu if line-45 formula is applied to the code-style init BEFORE iter 1:", mu_code)
    C0, M0, WD0, WC0, S0 = zero_init(X, Y)
    mu_zero = mu_from_code_formula(X, Y, WC0, WD0, S0)
    print("mu if line-45 formula is applied to zero W BEFORE iter 1:", mu_zero)

    # Paper sensitivity text: alpha in {1e-3,...,1e2}, gamma in {0.01,...,10},
    # beta=0.1 and eta=100 often, lambda_max=1e6. Mapped onto CODE field names.
    # These are probes inside the published ranges, not the unknown table optimum.
    paper_alpha = 1e-3  # multiplies -HSIC; code field `gamma`
    paper_gamma = 0.01  # nuclear norm; code field `alpha`
    paper_beta = 0.1    # individual product; code field `eta`
    paper_eta = 100.0   # ||W_D||_F^2; code field `beta`
    lam_at = theorem_lambda(paper_alpha, mu_code)
    print("Theorem1 lower bound at paper_alpha=1e-3 and code-init mu:", lam_at)
    lam_at_zero_mu = theorem_lambda(paper_alpha, mu_zero)
    print("Theorem1 lower bound at paper_alpha=1e-3 and zero-W mu:", lam_at_zero_mu)

    def pack(mu, lam):
        return {
            "mu": list(mu),
            "lambda": list(lam),
            "gamma": paper_alpha,
            "alpha": paper_gamma,
            "eta": paper_beta,
            "beta": paper_eta,
            "lambda_max": 1e6,
        }

    # 1. missing para entirely cannot start
    print("\n===== MISSING PARA (mirrors run.m) =====")
    print("skipped numeric call: Update_C indexes para.lambda / para.mu / para.gamma")

    # 2. code variable init, but lambda=mu=0 (para fields left at zero)
    one_c_step(
        "code_var_init + mu=0 + lambda=0",
        X, Y, code_style_init,
        pack([0.0, 0.0], [0.0, 0.0]),
    )
    # 3. zero variable init, lambda=mu=0
    one_c_step(
        "zero_var_init + mu=0 + lambda=0",
        X, Y, zero_init,
        pack([0.0, 0.0], [0.0, 0.0]),
    )
    # 4. code var init, mu from formula, lambda=0
    one_c_step(
        "code_var_init + mu=line45 + lambda=0",
        X, Y, code_style_init,
        pack(mu_code, [0.0, 0.0]),
    )
    # 5. code var init, mu from formula, lambda = half / at / 10x theorem bound
    one_c_step(
        "code_var_init + mu=line45 + lambda=0.5*theorem",
        X, Y, code_style_init,
        pack(mu_code, [0.5 * x for x in lam_at]),
    )
    one_c_step(
        "code_var_init + mu=line45 + lambda=theorem",
        X, Y, code_style_init,
        pack(mu_code, lam_at),
    )
    one_c_step(
        "code_var_init + mu=line45 + lambda=10*theorem",
        X, Y, code_style_init,
        pack(mu_code, [10 * x for x in lam_at]),
    )
    # 6. zero W so the HSIC Gram is 0, lambda at the zero-W theorem bound
    one_c_step(
        "zero_var_init + mu=line45(zero W) + lambda=theorem",
        X, Y, zero_init,
        pack(mu_zero, lam_at_zero_mu),
    )

    # short alternating loops
    short_loop(
        "code_var_init lambda=0 mu=line45",
        X, Y, code_style_init,
        pack(mu_code, [0.0, 0.0]),
        n_iter=3,
    )
    short_loop(
        "zero_var_init lambda=theorem mu=line45",
        X, Y, zero_init,
        pack(mu_zero, lam_at_zero_mu),
        n_iter=3,
    )
    short_loop(
        "code_var_init lambda=theorem mu=line45",
        X, Y, code_style_init,
        pack(mu_code, lam_at),
        n_iter=8,
    )


if __name__ == "__main__":
    main()
