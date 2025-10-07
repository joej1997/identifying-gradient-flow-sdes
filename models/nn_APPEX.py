from __future__ import annotations
import numpy as np
import equinox as eqx, jax.numpy as jnp, jax, optax
from utils.APPEX_helpers import _normalise_trajectory
from utils.MLE_parameter_estimation import fit_nn_drift, _estimate_sigma2_isotropic
from utils.SB_solvers import AEOT_trajectory_inference, MMOT_trajectory_inference
from typing import Callable, Optional
import os, json, csv
import time
import csv
import math
import pandas as pd
from typing import Callable, Sequence

def sample_paths_backward_log(K_list_log, log_u_list, N_paths, rng):
    T = len(log_u_list)
    # compute left messages once (same as in IPFP)
    def _lse(A, axis=-1):
        m = np.max(A, axis=axis, keepdims=True)
        return (m + np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True))).squeeze(axis)

    # left messages
    log_l = [None]*T
    log_l[0] = np.zeros_like(log_u_list[0])               # ℓ_1 = 1
    for i in range(1, T):
        log_l[i] = _lse(K_list_log[i-1] + log_u_list[i-1][:, None], axis=0)

    idxs = np.empty((N_paths, T), dtype=int)

    # sample x_T from log m_T = log u_T + log ℓ_T
    log_mT = log_u_list[-1] + log_l[-1]
    pT = np.exp(log_mT - np.max(log_mT)); pT /= pT.sum()
    idxs[:, -1] = rng.choice(pT.size, size=N_paths, p=pT)

    # go backward: P(x_i=j | x_{i+1}=k) ∝ u_i(j) ℓ_i(j) K_i(j,k)
    for i in range(T-2, -1, -1):
        logKi = K_list_log[i]
        log_ui = log_u_list[i]
        log_li = log_l[i]
        for n in range(N_paths):
            k = idxs[n, i+1]
            lcol = logKi[:, k] + log_ui + log_li     # (n_i,)
            lcol -= np.max(lcol)
            pcol = np.exp(lcol); s = pcol.sum()
            pcol = pcol/s if np.isfinite(s) and s>0 else np.full_like(pcol, 1.0/pcol.size)
            idxs[n, i] = rng.choice(pcol.size, p=pcol)
    return idxs

def sample_paths_from_scalings_log(
    K_list_log,    # [log K_1,..., log K_{T-1}]
    log_u_list,    # [log u_1,...,log u_T]
    N_paths: int,
    rng: np.random.Generator,
):
    """Sample indices (N_paths,T) without ever exponentiating large numbers."""
    T = len(log_u_list)
    idxs = np.empty((N_paths, T), dtype=int)

    # initial: p0 ∝ exp(log u_1 - max)
    l0 = log_u_list[0]
    l0c = l0 - np.max(l0)
    p0 = np.exp(l0c)
    p0 /= p0.sum()
    idxs[:, 0] = rng.choice(p0.size, size=N_paths, p=p0)

    for i in range(T - 1):
        logKi = K_list_log[i]
        log_ui1 = log_u_list[i+1]
        for n in range(N_paths):
            j = idxs[n, i]
            # log row ∝ log K_i[j,:] + log u_{i+1}
            lrow = logKi[j, :] + log_ui1
            lrow -= np.max(lrow)
            prow = np.exp(lrow)
            s = prow.sum()
            if not np.isfinite(s) or s <= 0:
                prow = np.ones_like(prow) / prow.size
            else:
                prow /= s
            idxs[n, i+1] = rng.choice(prow.size, p=prow)
    return idxs

def build_time_kernels(X_slices, dt, b_fn, Sigma):
    # returns (K_list, K_list_log)
    d = X_slices[0].shape[1]
    Sigma_dt = Sigma * dt
    chol = np.linalg.cholesky(Sigma_dt)
    inv = np.linalg.inv(Sigma_dt)
    log_norm = -0.5 * d * np.log(2*np.pi) - 0.5 * np.log(np.linalg.det(Sigma_dt))

    K_list, K_list_log = [], []
    for i in range(len(X_slices) - 1):
        X = X_slices[i]      # (n_i, d)
        Y = X_slices[i+1]    # (n_{i+1}, d)
        mu = X + _b_fn_batch(b_fn, X) * dt   # (n_i, d)
        # compute Gaussian kernel N(y; mu_x, Sigma dt)
        # log K[j,k] = log_norm - 0.5 (y_k - mu_j)^T inv (y_k - mu_j)
        diff = Y[None, :, :] - mu[:, None, :]
        quad = np.einsum('ijk,kl,ijl->ij', diff, inv, diff)
        logK = log_norm - 0.5 * quad
        # optional stabilization: subtract row-wise max before exp
        logK = logK - np.max(logK, axis=1, keepdims=True)
        K = np.exp(logK)
        # (no need to row-normalize; scalings u handle marginals)
        K_list.append(K)
        K_list_log.append(logK)
    return K_list, K_list_log

def _lse(A, axis=-1):
    m = np.max(A, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True))).squeeze(axis)

def unbalanced_mmot_ipfp_stable(
    K_list_log,   # [log K_1 (n1 x n2), ..., log K_{T-1} (n_{T-1} x n_T)]
    a_list,       # [a_1 (n1,), ..., a_T (nT,)]  (will be renormalized & floored)
    eps: float,   # entropic strength ε
    lam: float,   # data-fit λ
    max_iter: int = 500,
    tol: float = 1e-5,
    clip_update: float = 5.0,      # clip on (log a - log m)
    eta: float = 1.0,              # extra damping; use η∈(0,1], default 1
):
    """
    Unbalanced time-chain IPFP with KL penalties: log u_i <- log u_i + η τ clip(log a_i - log m_i).
    Center log u_i after each step to prevent overflow/underflow.
    """
    def _lse(A, axis=-1):
        m = np.max(A, axis=axis, keepdims=True)
        return (m + np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True))).squeeze(axis)

    T = len(a_list)
    assert len(K_list_log) == T - 1

    # normalize + floor targets
    a = [np.asarray(ai, float) for ai in a_list]
    for i in range(T):
        ai = np.maximum(a[i], 0.0)
        s = ai.sum()
        if not np.isfinite(s) or s <= 0:
            raise ValueError(f"a[{i}] has nonpositive sum.")
        a[i] = np.maximum(ai / s, 1e-300)
    log_a = [np.log(ai) for ai in a]

    # init scalings
    log_u = [np.zeros_like(ai) for ai in a]
    tau = 1.0 / (1.0 + lam / eps)           # (0,1]
    step = float(eta) * float(tau)

    def right_messages(log_u):
        # log v_i[j] = logsumexp_k( log K_i[j,k] + log u_{i+1}[k] )
        log_v = [None] * T
        log_v[-1] = np.zeros_like(log_u[-1])     # v_T = 1
        for i in range(T - 2, -1, -1):
            log_v[i] = _lse(K_list_log[i] + log_u[i+1][None, :], axis=1)
        return log_v

    def left_messages(log_u, log_v):
        # log ℓ_i[k] = logsumexp_j( log K_{i-1}[j,k] + log v_{i-1}[j] )
        log_l = [None] * T
        log_l[0] = np.zeros_like(log_u[0])       # ℓ_1 = 1
        for i in range(1, T):
            log_l[i] = _lse(K_list_log[i-1] + log_v[i-1][:, None], axis=0)
        return log_l

    def center(x):
        # subtract logsumexp to keep numbers moderate
        m = _lse(x, axis=0)  # scalar
        return x - m

    prev_res = np.inf
    for _ in range(max_iter):
        log_v = right_messages(log_u)
        log_l = left_messages(log_u, log_v)
        log_m = [log_u[i] + log_l[i] + log_v[i] for i in range(T)]

        # sup-norm residual
        res = max(float(np.max(np.abs(log_a[i] - log_m[i]))) for i in range(T))

        # damped, clipped update + centering
        for i in range(T):
            delta = log_a[i] - log_m[i]
            if clip_update is not None:
                delta = np.clip(delta, -clip_update, clip_update)
            log_u[i] = log_u[i] + step * delta
            log_u[i] = center(log_u[i])

        if step * res < tol or abs(prev_res - res) < tol * 0.1:
            break
        prev_res = res

    # final messages/marginals
    log_v = right_messages(log_u)
    log_l = left_messages(log_u, log_v)
    log_m = [log_u[i] + log_l[i] + log_v[i] for i in range(T)]
    return log_u, log_m

import numpy as np
from typing import Callable, Sequence, Optional, Tuple, Dict, Any


def _b_fn_batch(b_fn: Callable[[np.ndarray], np.ndarray], Z: np.ndarray) -> np.ndarray:
    """Call drift on a batch (tolerates vectorized or per-point implementations)."""
    Z = np.asarray(Z, float)
    try:
        out = np.asarray(b_fn(Z), float)
        if out.shape == Z.shape:
            return out
    except Exception:
        pass
    if Z.ndim == 1:
        return np.asarray(b_fn(Z), float)
    return np.vstack([np.asarray(b_fn(z), float) for z in Z])


# ---------- small numeric utils ----------
def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)

def sb_refine(
        model=None,
        state=None,
        eval_dataset=None,
        n_outer: int = 30, # set to 1 for the non-iterative WOT algorithm
        n_traj_sample: int = 2000,
        fix_diffusion: bool = False, # True for naive SBIRR, which fixes the reference diffusion
        init_sigma2: float = 1.0,
        init_drift: Callable[[np.ndarray], np.ndarray] | None = None,
        solver: str = "mmot",
        lambda_df: float = 1.0,
        smc_particles: int = 8192,
        kde_h: float | None = None,
        # --- NN hyper-parameters
        nn_width: int = 128,
        nn_depth: int = 2,
        nn_lr: float = 3e-3,
        nn_epochs: int = 500,
        nn_conservative: bool = True,
        nn_activation: str | None = None,
        save_dir: Optional[str] = None,
        generative: bool = False
) -> tuple:
    """
    Returns (drift_fn, theta_or_model, sigma2) or (..., trace) if return_trace.

    This version augments the loop with a KL-based surrogate (unchanged)
    and **per-iteration timing** of:
      - Trajectory inference
      - Drift MLE
      - Diffusion MLE (σ² update)

    Two CSVs (if save_dir is provided):
      * sb_timing_iter.csv: per-iteration times
      * sb_timing_summary.csv: mean/std/sem per phase
    """
    assert eval_dataset is not None, "sb_refine needs eval_dataset"
    dt = float(eval_dataset.dt)
    d = int(eval_dataset.data_dim)

    # ------------- measured snapshots (N, T, d) -------------
    X_meas = _normalise_trajectory(eval_dataset)  # (N, T, d)
    N_meas, T_meas, _ = X_meas.shape

    # ------------- initial/reference drift ------------------
    if init_drift is not None:
        drift_ref = init_drift
    elif model is not None and state is not None:
        phi_hat = model.get_potential(state)
        def _single_grad(z): return -jax.grad(lambda u: phi_hat(u))(z)
        def drift_ref(X):
            Xj = jnp.asarray(X)
            out = _single_grad(Xj) if Xj.ndim == 1 else jax.vmap(_single_grad)(Xj)
            return np.asarray(out)
    else:
        drift_ref = lambda X: np.zeros_like(X)

    # current iterate
    sigma2 = float(init_sigma2)
    Sigma = sigma2 * np.eye(d)
    theta_or_model, drift_fn = None, drift_ref
    print(f"[SB] initial estimated diffusivity (σ²): {sigma2:.6g}")
    trace: list[dict] = []  # per-iteration metrics

    # NEW: timing containers
    timing_rows: list[dict] = []  # one row per iteration

    # ===================== outer iterations =====================
    for it in range(n_outer):
        print(f"[SB]  outer iter {it + 1}/{n_outer}")

        # ------------------ (NEW) drift-only augmentation (no diffusion) ------------------
        t_gen0 = time.perf_counter()

        if generative:
            print('not yet supported')
        else:
            X_aug_rect = X_meas

        t_gen1 = time.perf_counter()
        t_gen = t_gen1 - t_gen0
        # -------------------------------------------------------------------------------

        # ------------------ E-step: infer trajectories ------------------
        t0 = time.perf_counter()
        if solver.lower() in {"mmot", "ipfp"}:
            X_OT, P_list = MMOT_trajectory_inference(
                X_aug_rect, dt, est_A=drift_fn, est_Sigma=Sigma,
                max_iter=200, tol=1e-5, N_sample_traj=n_traj_sample, use_log_domain=True
            )
        elif solver.lower() in {"unbalanced", "unbalanced_ipfp"}:
            # supports per slice
            X_list = [np.asarray(X_aug_rect[:, i, :], float) for i in range(T_meas)]
            # relaxed targets (uniform on supports is fine; or plug KDE weights)
            a_list = [np.ones(x.shape[0], float) / max(1, x.shape[0]) for x in X_list]

            # time kernels from the reference SDE
            K_list, K_list_log = build_time_kernels(X_list, dt, drift_fn, Sigma)

            # stable unbalanced IPFP (all log-domain)
            log_u, log_m = unbalanced_mmot_ipfp_stable(
                K_list_log, a_list,
                eps=1.0,  # entropic strength ε
                lam=lambda_df,  # data-fit λ   -> τ = 1/(1+λ/ε)
                max_iter=300, tol=1e-4,
                clip_update=5.0,  # keep steps bounded
                eta=0.9  # extra damping; raise toward 1.0 if stable
            )

            # sample indices in log-domain (no exp!)
            rng = np.random.default_rng(1234)
            idxs = sample_paths_backward_log(
                K_list_log, log_u, N_paths=n_traj_sample, rng=rng
            )

            # lift indices to coordinates
            X_OT = np.stack([X_list[t][idxs[:, t], :] for t in range(T_meas)], axis=1)
        else:
            raise ValueError(f"Unknown solver: {solver}")
        t1 = time.perf_counter()
        t_traj = t1 - t0

        # ------------------ M-step: fit drift ---------------------------
        t2 = time.perf_counter()
        fit_kwargs = dict(
            X=X_OT, dt=dt,
            key=jax.random.PRNGKey(it),
            width=nn_width, depth=nn_depth,
            lr=nn_lr, n_epochs=nn_epochs,
            conservative=nn_conservative
        )
        if nn_activation is not None:
            fit_kwargs["activation"] = nn_activation
        try:
            drift_new, nn_model = fit_nn_drift(**fit_kwargs)
        except TypeError:
            fit_kwargs.pop("activation", None)
            drift_new, nn_model = fit_nn_drift(**fit_kwargs)
        theta_or_model = nn_model
        t3 = time.perf_counter()
        t_drift = t3 - t2

        # ------------------ σ² update -----------------------------------
        t4 = time.perf_counter()  # NEW: start diffusion MLE timer
        if fix_diffusion:
            sigma2_new = sigma2
        else:
            sigma2_new = _estimate_sigma2_isotropic(X_OT, dt, drift_new)
        t5 = time.perf_counter()
        t_sigma2 = t5 - t4  # NEW

        print(f"[SB]  iter {it+1} estimated diffusivity (σ²): {sigma2_new:.6g}")

        Sigma = max(float(sigma2_new), 1e-8) * np.eye(d)
        sigma2 = float(Sigma[0, 0])

        # accept parameters
        drift_prev = drift_fn
        drift_fn = drift_new

        timing_rows.append({
            "iter": it + 1,
            "t_gen_s": t_gen,  # <-- NEW
            "t_traj_s": t_traj,
            "t_drift_s": t_drift,
            "t_sigma2_s": t_sigma2,
            "t_total_s": (t_gen + t_traj + t_drift + t_sigma2),  # include gen
            "nn_width": nn_width,
            "nn_depth": nn_depth,
            "nn_epochs": nn_epochs,
            "n_traj_sample": n_traj_sample,
            "dt": dt,
            "data_dim": d,
            "fix_diffusion": bool(fix_diffusion),
        })

    # --------- NEW: save timing CSVs ----------
    if save_dir is not None and len(timing_rows) > 0:
        os.makedirs(save_dir, exist_ok=True)
        iter_path = os.path.join(save_dir, "sb_timing_iter.csv")
        summary_path = os.path.join(save_dir, "sb_timing_summary.csv")
        _save_timing_csvs(timing_rows, iter_path, summary_path)

    return (drift_fn, theta_or_model, sigma2)




def _save_timing_csvs(timing_rows: list[dict], iter_csv_path: str, summary_csv_path: str) -> None:
    """Write per-iteration timing and an across-iteration summary (mean/std/sem)."""
    # Per-iteration
    if pd is not None:
        df = pd.DataFrame(timing_rows)
        df.to_csv(iter_csv_path, index=False)
        # Summary
        phases = ["t_gen_s", "t_traj_s", "t_drift_s", "t_sigma2_s", "t_total_s"]  # <-- added t_gen_s
        stats = []
        for ph in phases:
            vals = df[ph].to_numpy(dtype=float)
            n = int(np.isfinite(vals).sum())
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                mean = std = sem = float("nan")
                n_eff = 0
            else:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                sem = float(std / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
                n_eff = len(vals)
            stats.append({"phase": ph, "mean_s": mean, "std_s": std, "sem_s": sem, "n": n_eff})
        pd.DataFrame(stats).to_csv(summary_csv_path, index=False)
    else:
        # Fallback: stdlib csv
        fieldnames = sorted({k for row in timing_rows for k in row.keys()})
        with open(iter_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in timing_rows:
                w.writerow(r)
        # Summary with minimal deps
        phases = ["t_traj_s", "t_drift_s", "t_sigma2_s", "t_total_s"]
        with open(summary_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["phase", "mean_s", "std_s", "sem_s", "n"])
            w.writeheader()
            for ph in phases:
                vals = [float(r[ph]) for r in timing_rows if r.get(ph) is not None]
                n = len(vals)
                if n == 0:
                    w.writerow({"phase": ph, "mean_s": float("nan"), "std_s": float("nan"),
                                "sem_s": float("nan"), "n": 0})
                    continue
                mean = sum(vals) / n
                if n > 1:
                    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
                    std = math.sqrt(var)
                    sem = std / math.sqrt(n)
                else:
                    std = 0.0
                    sem = 0.0
                w.writerow({"phase": ph, "mean_s": mean, "std_s": std, "sem_s": sem, "n": n})

