from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from r1_core import (
    _lag_cube, _loo_mean, adjust_pvalues, linear_all, stable_seed,
    unit_linear_edge,
)

def circular_shift_matrix(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.array(X, copy=True)
    T, N, K = out.shape
    for i in range(N):
        for k in range(K):
            shift = int(rng.integers(1, T))
            out[:, i, k] = np.roll(out[:, i, k], shift)
    return out


def block_permute_vector(v: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(v)
    starts = list(range(0, n, block))
    blocks = [v[s:min(s + block, n)].copy() for s in starts]
    rng.shuffle(blocks)
    return np.concatenate(blocks)[:n]


def block_permute_matrix(X: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    out = np.array(X, copy=True)
    T, N, K = out.shape
    for i in range(N):
        for k in range(K):
            out[:, i, k] = block_permute_vector(out[:, i, k], block, rng)
    return out


def _forward_splits(times: np.ndarray, fractions: Iterable[float], horizon: int, min_train: int) -> list[tuple[np.ndarray, np.ndarray]]:
    u = np.array(sorted(np.unique(times)))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for frac in fractions:
        cut = max(min_train, int(math.floor(len(u) * float(frac))))
        if cut >= len(u):
            continue
        test = u[cut:min(cut + horizon, len(u))]
        if len(test) == 0:
            continue
        splits.append((u[:cut], test))
    seen = set()
    unique_splits = []
    for tr, te in splits:
        key = (tuple(tr), tuple(te))
        if key not in seen:
            seen.add(key)
            unique_splits.append((tr, te))
    return unique_splits


def _rf_design(Y: np.ndarray, x: np.ndarray, max_lag: int, include_cce: bool = True) -> tuple[pd.DataFrame, list[str], list[str]]:
    p = max_lag
    y = Y[p:, :]
    yl = _lag_cube(Y, p)
    xl = _lag_cube(x, p)
    L, N = y.shape
    times = np.repeat(np.arange(p, p + L), N)
    units = np.tile(np.arange(N), L)
    data: dict[str, Any] = {"y": y.reshape(-1), "time": times, "unit": units.astype(str)}
    y_terms, x_terms, control_terms = [], [], []
    for j in range(p):
        yn = f"y_lag{j+1}"
        xn = f"x_lag{j+1}"
        data[yn] = yl[:, :, j].reshape(-1)
        data[xn] = xl[:, :, j].reshape(-1)
        y_terms.append(yn)
        x_terms.append(xn)
    if include_cce and N > 1:
        cy = _loo_mean(yl)
        cx = _loo_mean(xl)
        for j in range(p):
            cyn = f"mean_y_lag{j+1}"
            cxn = f"mean_x_lag{j+1}"
            data[cyn] = cy[:, :, j].reshape(-1)
            data[cxn] = cx[:, :, j].reshape(-1)
            control_terms.extend([cyn, cxn])
    tt = (times - times.min()) / max(times.max() - times.min(), 1)
    data["time_scaled"] = tt
    control_terms.append("time_scaled")
    df = pd.DataFrame(data)
    unit_dummies = pd.get_dummies(df["unit"], prefix="unit", drop_first=True, dtype=float)
    df = pd.concat([df.drop(columns=["unit"]), unit_dummies], axis=1)
    control_terms.extend(unit_dummies.columns.tolist())
    return df, y_terms + control_terms, x_terms


def _rf_delta(Y: np.ndarray, x: np.ndarray, cfg: dict[str, Any], seed: int, trees: int | None = None,
              include_cce: bool = True, fractions: list[float] | None = None) -> tuple[np.ndarray, float]:
    rcfg = cfg["rf"]
    trees = int(trees if trees is not None else rcfg["trees_mc"])
    fractions = fractions if fractions is not None else list(rcfg["forward_fractions"])
    df, restricted, x_terms = _rf_design(Y, x, int(cfg["max_lag"]), include_cce=include_cce)
    full = restricted + x_terms
    splits = _forward_splits(df["time"].to_numpy(), fractions, int(rcfg["horizon"]), int(rcfg["min_train_times"]))
    deltas: list[float] = []
    for fold, (tr_t, te_t) in enumerate(splits):
        tr = df["time"].isin(tr_t).to_numpy()
        te = df["time"].isin(te_t).to_numpy()
        if int(tr.sum()) < 10 or int(te.sum()) < 2:
            continue
        params = dict(
            n_estimators=trees,
            max_depth=int(rcfg["max_depth"]),
            min_samples_leaf=int(rcfg["min_node_size"]),
            max_features="sqrt",
            random_state=seed + 1009 * (fold + 1),
            n_jobs=1,
        )
        m0 = RandomForestRegressor(**params).fit(df.loc[tr, restricted], df.loc[tr, "y"])
        params["random_state"] = seed + 1009 * (fold + 1) + 500000
        m1 = RandomForestRegressor(**params).fit(df.loc[tr, full], df.loc[tr, "y"])
        mse0 = mean_squared_error(df.loc[te, "y"], m0.predict(df.loc[te, restricted]))
        mse1 = mean_squared_error(df.loc[te, "y"], m1.predict(df.loc[te, full]))
        if np.isfinite(mse0) and mse0 > 1e-12 and np.isfinite(mse1):
            deltas.append(100.0 * (mse0 - mse1) / mse0)
    arr = np.asarray(deltas, dtype=float)
    return arr, float(np.mean(arr)) if len(arr) else math.nan


def rf_edge(Y: np.ndarray, x: np.ndarray, cfg: dict[str, Any], seed: int, shadow_repeats: int | None = None,
            trees: int | None = None, shadow_type: str = "circular", block: int = 3,
            include_cce: bool = True) -> dict[str, Any]:
    rcfg = cfg["rf"]
    B = int(shadow_repeats if shadow_repeats is not None else rcfg["shadow_repeats_mc"])
    real, real_mean = _rf_delta(Y, x, cfg, seed, trees=trees, include_cce=include_cce)
    if not len(real) or not np.isfinite(real_mean):
        return {"rf_mean": math.nan, "rf_median": math.nan, "rf_fold_share": math.nan,
                "rf_shadow_q95": math.nan, "rf_emp_p": 1.0, "rf_supported": False,
                "rf_n_folds": 0}
    shadow = []
    rng = np.random.default_rng(seed + 7717)
    raw = x[:, :, None]
    for b in range(B):
        if shadow_type == "circular":
            sx = circular_shift_matrix(raw, rng)[:, :, 0]
        elif shadow_type == "block":
            sx = block_permute_matrix(raw, block, rng)[:, :, 0]
        else:
            raise ValueError(shadow_type)
        _, d = _rf_delta(Y, sx, cfg, seed + 100000 * (b + 1), trees=trees, include_cce=include_cce)
        if np.isfinite(d):
            shadow.append(d)
    sh = np.asarray(shadow, dtype=float)
    q95 = float(np.quantile(sh, 0.95)) if len(sh) else math.nan
    p_emp = float((1 + np.sum(sh >= real_mean)) / (len(sh) + 1)) if len(sh) else 1.0
    fold_share = float(np.mean(real > 0))
    supported = bool(
        np.isfinite(real_mean) and real_mean > 0 and np.isfinite(q95) and real_mean > q95
        and fold_share >= float(rcfg["positive_fold_share"])
    )
    return {
        "rf_mean": float(real_mean),
        "rf_median": float(np.median(real)),
        "rf_fold_share": fold_share,
        "rf_shadow_q95": q95,
        "rf_emp_p": p_emp,
        "rf_supported": supported,
        "rf_n_folds": int(len(real)),
    }


def unit_replication_counts(Y: np.ndarray, X: np.ndarray, pooled: pd.DataFrame, rf_rows: list[dict[str, Any]],
                            cfg: dict[str, Any], seed: int) -> np.ndarray:
    T, N = Y.shape
    K = X.shape[2]
    alpha = float(cfg["alpha"])
    logbf_thr = float(cfg["bayesian"]["log_bf_threshold"])

    classical_counts = np.zeros(K, dtype=int)
    bayes_counts = np.zeros(K, dtype=int)
    for i in range(N):
        p = np.ones(K, dtype=float)
        lb = np.full(K, -100.0)
        for k in range(K):
            r = unit_linear_edge(Y[:, i], X[:, i, k], int(cfg["max_lag"]))
            p[k] = r["p_lag"]
            lb[k] = r["logbf_max"]
        _, qby = adjust_pvalues(p)
        classical_counts += (qby <= float(cfg["fdr_q"])).astype(int)
        bayes_counts += (lb >= logbf_thr).astype(int)

    rf_counts = np.zeros(K, dtype=int)
    min_rep = int(cfg["tiering"]["minimum_replications"])
    for k in range(K):
        if not bool(rf_rows[k]["rf_supported"]):
            continue
        count = 0
        for i in range(N):
            yy = Y[:, i:i+1]
            xx = X[:, i:i+1, k]
            r = rf_edge(
                yy, xx, cfg, stable_seed(seed, "rf-unit", k, i),
                shadow_repeats=max(9, min(19, int(cfg["rf"]["shadow_repeats_mc"]))),
                trees=max(50, min(100, int(cfg["rf"]["trees_mc"]))),
                include_cce=False,
            )
            if r["rf_supported"]:
                count += 1
                if count >= min_rep:
                    break
        rf_counts[k] = count

    return np.maximum(np.maximum(classical_counts, bayes_counts), rf_counts)


def placebo_gate(Y: np.ndarray, X: np.ndarray, observed_linear: pd.DataFrame, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    B = int(cfg["tiering"]["placebo_repeats"])
    threshold = float(cfg["tiering"]["placebo_quantile"])
    q = float(cfg["fdr_q"])
    observed_count = int(np.sum(observed_linear["q_by"].to_numpy() <= q))
    rng = np.random.default_rng(seed + 8811)
    counts = []
    for b in range(B):
        shifted = circular_shift_matrix(X, rng)
        lin = linear_all(Y, shifted, cfg, control="time_fe")
        counts.append(int(np.sum(lin["q_by"].to_numpy() <= q)))
    q95 = float(np.quantile(counts, threshold)) if counts else math.nan
    empirical_p = float((1 + np.sum(np.asarray(counts) >= observed_count)) / (len(counts) + 1)) if counts else 1.0
    return {
        "placebo_observed_count": observed_count,
        "placebo_q95_count": q95,
        "placebo_emp_p": empirical_p,
        "placebo_pass": bool(observed_count > q95) if np.isfinite(q95) else False,
    }


def assign_tier(classical: bool, bayes: bool, rf: bool, replication_count: int, placebo_pass: bool,
                sensitivity_pass: bool, grouped: bool = False) -> tuple[str, int]:
    replicated = replication_count >= 2
    if grouped:
        linear = bool(classical or bayes)
        n_families = int(linear) + int(rf)
        if n_families >= 2 and replicated and placebo_pass and sensitivity_pass:
            return "triangulated temporal precedence", n_families
        if rf and not linear:
            return "nonlinear candidate; replication required", n_families
        if n_families >= 1 and replicated:
            return "replicated single-family evidence", n_families
        if n_families >= 1:
            return "exploratory evidence", n_families
        return "unsupported", 0
    n_methods = int(classical) + int(bayes) + int(rf)
    if n_methods >= 2 and replicated and placebo_pass and sensitivity_pass:
        return "triangulated temporal precedence", n_methods
    if rf and not classical and not bayes:
        return "nonlinear candidate; replication required", n_methods
    if n_methods >= 1 and replicated:
        return "replicated single-family evidence", n_methods
    if n_methods >= 1:
        return "exploratory evidence", n_methods
    return "unsupported", 0
