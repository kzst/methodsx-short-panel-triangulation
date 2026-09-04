from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import f as fdist
from statsmodels.stats.multitest import multipletests


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def stable_seed(base: int, *parts: Any) -> int:
    text = "|".join(map(str, (base, *parts))).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def ar1(rng: np.random.Generator, n: int, phi: float, sigma: float = 1.0) -> np.ndarray:
    out = np.zeros(n, dtype=float)
    sd0 = sigma / max(math.sqrt(max(1.0 - phi * phi, 1e-6)), 0.25)
    out[0] = rng.normal(scale=sd0)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + rng.normal(scale=sigma)
    return out


def _standardize_by_unit(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        mu = np.nanmean(x, axis=0, keepdims=True)
        sd = np.nanstd(x, axis=0, ddof=1, keepdims=True)
    else:
        mu = np.nanmean(x, axis=0, keepdims=True)
        sd = np.nanstd(x, axis=0, ddof=1, keepdims=True)
    sd = np.where(np.isfinite(sd) & (sd > 1e-10), sd, 1.0)
    return (x - mu) / sd


@dataclass(frozen=True)
class Cell:
    cell_id: str
    scenario: str
    reps: int
    T: int
    N: int
    K: int
    phi_x: float
    phi_y: float
    cross_dependence: float
    linear_effect: float
    nonlinear_effect: float
    n_true_linear: int
    n_true_nonlinear: int
    correlated_predictor_blocks: bool = True
    family: str = "primary"


def simulate_panel(cell: Cell, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate T x N outcome and T x N x K predictors with known truth."""
    rng = np.random.default_rng(seed)
    T, N, K = cell.T, cell.N, cell.K
    rho = float(np.clip(cell.cross_dependence, 0.0, 0.95))

    global_factor = ar1(rng, T, 0.60, 1.0)
    n_blocks = max(1, math.ceil(K / 5))
    block_factors = np.column_stack([ar1(rng, T, 0.55, 1.0) for _ in range(n_blocks)])

    X = np.zeros((T, N, K), dtype=float)
    for k in range(K):
        b = min(k // 5, n_blocks - 1)
        shared = 0.55 * global_factor + 0.45 * block_factors[:, b]
        for i in range(N):
            innov = math.sqrt(rho) * shared + math.sqrt(max(1.0 - rho, 1e-8)) * rng.normal(size=T)
            x = np.zeros(T, dtype=float)
            x[0] = innov[0]
            for t in range(1, T):
                x[t] = cell.phi_x * x[t - 1] + innov[t]
            X[:, i, k] = x

    if cell.correlated_predictor_blocks and K > 1:
        for b in range(n_blocks):
            idx = np.arange(b * 5, min((b + 1) * 5, K))
            if len(idx) > 1:
                block_mean = np.mean(X[:, :, idx], axis=2, keepdims=True)
                X[:, :, idx] = 0.80 * X[:, :, idx] + 0.20 * block_mean

    X = _standardize_by_unit(X)
    truth = np.zeros(K, dtype=bool)
    nlin = min(cell.n_true_linear, K)
    nnon = min(cell.n_true_nonlinear, max(0, K - nlin))
    if cell.scenario != "global_null":
        truth[:nlin] = True
        truth[nlin:nlin + nnon] = True

    unit_intercept = rng.normal(scale=0.60, size=N)
    unit_loading = rng.normal(loc=0.60, scale=0.12, size=N)
    Y = np.zeros((T, N), dtype=float)
    Y[0, :] = unit_intercept + unit_loading * global_factor[0] + rng.normal(scale=1.0, size=N)

    lin_scale = max(math.sqrt(max(nlin, 1)), 1.0)
    non_scale = max(math.sqrt(max(nnon, 1)), 1.0)
    for t in range(1, T):
        for i in range(N):
            direct = 0.0
            if cell.scenario != "global_null":
                if nlin:
                    direct += cell.linear_effect * float(np.sum(X[t - 1, i, :nlin])) / lin_scale
                if nnon:
                    z = X[t - 1, i, nlin:nlin + nnon]
                    direct += cell.nonlinear_effect * float(np.sum(z * z - 1.0)) / non_scale
            Y[t, i] = (
                unit_intercept[i]
                + cell.phi_y * Y[t - 1, i]
                + direct
                + math.sqrt(rho) * unit_loading[i] * global_factor[t]
                + rng.normal(scale=1.0)
            )

    return Y, X, truth


def _two_way_demean(a: np.ndarray) -> np.ndarray:
    if a.ndim == 2:
        return a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    return a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean(axis=(0, 1), keepdims=True)


def _unit_demean(a: np.ndarray) -> np.ndarray:
    return a - a.mean(axis=0, keepdims=True)


def _loo_mean(a: np.ndarray) -> np.ndarray:
    n = a.shape[1]
    if n <= 1:
        return np.full_like(a, np.nan)
    total = np.sum(a, axis=1, keepdims=True)
    return (total - a) / (n - 1)


def _lag_cube(a: np.ndarray, p: int) -> np.ndarray:
    T, N = a.shape
    return np.stack([a[p - lag:T - lag, :] for lag in range(1, p + 1)], axis=2)


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    return beta, resid, rss, int(rank)


def _cluster_wald(y: np.ndarray, X: np.ndarray, units: np.ndarray, tested_idx: np.ndarray) -> tuple[float, float]:
    beta, resid, _, rank = _ols(y, X)
    n, k = X.shape
    groups = np.unique(units)
    G = len(groups)
    bread = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k), dtype=float)
    for g in groups:
        idx = units == g
        score = X[idx, :].T @ resid[idx]
        meat += np.outer(score, score)
    corr = 1.0
    if G > 1 and n > rank:
        corr = (G / (G - 1.0)) * ((n - 1.0) / (n - rank))
    V = corr * bread @ meat @ bread
    b = beta[tested_idx]
    Vs = V[np.ix_(tested_idx, tested_idx)]
    q = len(tested_idx)
    if q == 0:
        return 1.0, 0.0
    try:
        W = float(b.T @ np.linalg.pinv(Vs) @ b)
    except Exception:
        return 1.0, 0.0
    F = max(W / q, 0.0)
    p = float(fdist.sf(F, q, max(G - 1, 1)))
    return p, F


def _nested_f_pvalue(y: np.ndarray, X0: np.ndarray, X1: np.ndarray, q: int) -> tuple[float, float]:
    _, _, rss0, _ = _ols(y, X0)
    _, _, rss1, rank1 = _ols(y, X1)
    n = len(y)
    df2 = max(n - rank1, 1)
    if rss1 <= 0:
        return 1.0, 0.0
    F = max(((rss0 - rss1) / max(q, 1)) / max(rss1 / df2, 1e-14), 0.0)
    return float(fdist.sf(F, max(q, 1), df2)), F


def _bic(rss: float, n: int, rank: int) -> float:
    return n * math.log(max(rss / max(n, 1), 1e-14)) + rank * math.log(max(n, 2))


def pooled_linear_edge(Y: np.ndarray, x: np.ndarray, max_lag: int, control: str = "time_fe") -> dict[str, float]:
    pvals: list[float] = []
    logbfs: list[float] = []
    fstats: list[float] = []
    for p in range(1, max_lag + 1):
        y = Y[p:, :]
        yl = _lag_cube(Y, p)
        xl = _lag_cube(x, p)
        L, N = y.shape
        unit_ids = np.tile(np.arange(N), L)
        if control == "time_fe":
            yv = _two_way_demean(y).reshape(-1)
            YL = _two_way_demean(yl).reshape(-1, p)
            XL = _two_way_demean(xl).reshape(-1, p)
            X0 = YL
            X1 = np.column_stack([YL, XL])
        elif control == "cce":
            loo_y = _loo_mean(yl)
            loo_x = _loo_mean(xl)
            yv = _unit_demean(y).reshape(-1)
            YL = _unit_demean(yl).reshape(-1, p)
            XL = _unit_demean(xl).reshape(-1, p)
            CY = _unit_demean(loo_y).reshape(-1, p)
            CX = _unit_demean(loo_x).reshape(-1, p)
            X0 = np.column_stack([YL, CY, CX])
            X1 = np.column_stack([YL, CY, CX, XL])
        else:
            raise ValueError(control)

        pval, fstat = _cluster_wald(yv, X1, unit_ids, np.arange(X1.shape[1] - p, X1.shape[1]))
        _, _, rss0, r0 = _ols(yv, X0)
        _, _, rss1, r1 = _ols(yv, X1)
        logbf = 0.5 * (_bic(rss0, len(yv), r0) - _bic(rss1, len(yv), r1))
        pvals.append(pval)
        fstats.append(fstat)
        logbfs.append(float(np.clip(logbf, -100.0, 100.0)))

    finite = np.isfinite(pvals)
    if not any(finite):
        return {"p_min": 1.0, "p_lag": 1.0, "best_lag": np.nan, "f_max": 0.0, "logbf_max": -100.0}
    p_arr = np.asarray(pvals, dtype=float)
    best = int(np.nanargmin(p_arr))
    pmin = float(p_arr[best])
    p_lag = min(1.0, pmin * int(np.sum(finite)))
    return {
        "p_min": pmin,
        "p_lag": p_lag,
        "best_lag": best + 1,
        "f_max": float(fstats[best]),
        "logbf_max": float(np.nanmax(logbfs)),
    }


def unit_linear_edge(y: np.ndarray, x: np.ndarray, max_lag: int) -> dict[str, float]:
    pvals: list[float] = []
    logbfs: list[float] = []
    for p in range(1, max_lag + 1):
        yy = y[p:]
        yl = np.column_stack([y[p - lag:len(y) - lag] for lag in range(1, p + 1)])
        xl = np.column_stack([x[p - lag:len(x) - lag] for lag in range(1, p + 1)])
        X0 = np.column_stack([np.ones(len(yy)), yl])
        X1 = np.column_stack([X0, xl])
        pv, _ = _nested_f_pvalue(yy, X0, X1, p)
        _, _, rss0, r0 = _ols(yy, X0)
        _, _, rss1, r1 = _ols(yy, X1)
        pvals.append(pv)
        logbfs.append(float(np.clip(0.5 * (_bic(rss0, len(yy), r0) - _bic(rss1, len(yy), r1)), -100, 100)))
    pmin = float(np.min(pvals)) if pvals else 1.0
    return {"p_lag": min(1.0, pmin * max(len(pvals), 1)), "logbf_max": max(logbfs) if logbfs else -100.0}


def adjust_pvalues(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    q_bh = np.full_like(p, np.nan)
    q_by = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if np.any(ok):
        q_bh[ok] = multipletests(p[ok], method="fdr_bh")[1]
        q_by[ok] = multipletests(p[ok], method="fdr_by")[1]
    return q_bh, q_by


def linear_all(Y: np.ndarray, X: np.ndarray, cfg: dict[str, Any], control: str = "time_fe") -> pd.DataFrame:
    rows = []
    for k in range(X.shape[2]):
        r = pooled_linear_edge(Y, X[:, :, k], int(cfg["max_lag"]), control=control)
        r["k"] = k
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    qbh, qby = adjust_pvalues(out["p_lag"].to_numpy())
    out["q_bh"] = qbh
    out["q_by"] = qby
    return out
