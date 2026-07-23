from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.stats.multitest import multipletests


REPO_ROOT = Path(__file__).resolve().parents[1]


def ar1(rng: np.random.Generator, t: int, phi: float = 0.5, sigma: float = 1.0) -> np.ndarray:
    out = np.zeros(t, dtype=float)
    out[0] = rng.normal(scale=sigma / max(math.sqrt(1 - phi * phi), 0.25))
    for i in range(1, t):
        out[i] = phi * out[i - 1] + rng.normal(scale=sigma)
    return out


def simulate_panel(
    scenario: str,
    n_units: int = 24,
    n_time: int = 28,
    seed: int = 20260723,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a short panel with known temporal-precedence structure."""
    rng = np.random.default_rng(seed)
    common = ar1(rng, n_time, phi=0.65, sigma=0.65)
    time = np.arange(1, n_time + 1)
    rows: list[dict[str, float | int | str]] = []

    for u in range(n_units):
        unit_intercept = rng.normal(scale=0.7)
        loading = rng.normal(loc=0.55, scale=0.12)
        x_linear = ar1(rng, n_time, 0.45, 0.85) + loading * common + rng.normal(scale=0.10, size=n_time)
        x_nonlinear = ar1(rng, n_time, 0.35, 0.85) + 0.25 * common
        x_null = ar1(rng, n_time, 0.40, 0.95)
        x_common = 1.05 * common + ar1(rng, n_time, 0.25, 0.35)
        y = np.zeros(n_time, dtype=float)
        y[0] = unit_intercept + 0.7 * common[0] + rng.normal(scale=0.9)

        for t in range(1, n_time):
            direct = 0.0
            if scenario in {"mixed", "linear"}:
                direct += 0.75 * x_linear[t - 1]
            if scenario in {"mixed", "nonlinear"}:
                direct += 0.62 * (x_nonlinear[t - 1] ** 2 - 1.0)
            if scenario == "break" and t >= n_time // 2:
                direct += 0.95 * x_linear[t - 1]
            # In the common-shock scenario there is no x -> y path.
            y[t] = (
                unit_intercept
                + 0.42 * y[t - 1]
                + direct
                + 0.72 * common[t]
                + rng.normal(scale=0.90)
            )

        for t in range(n_time):
            rows.append(
                {
                    "unit": f"U{u + 1:02d}",
                    "time": int(time[t]),
                    "y": float(y[t]),
                    "x_linear": float(x_linear[t]),
                    "x_nonlinear": float(x_nonlinear[t]),
                    "x_null": float(x_null[t]),
                    "x_common": float(x_common[t]),
                    "common_factor": float(common[t]),
                    "scenario": scenario,
                }
            )

    df = pd.DataFrame(rows)
    truth = pd.DataFrame(
        [
            {
                "variable": "x_linear",
                "true_precedence": scenario in {"mixed", "linear", "break"},
                "functional_form": "linear" if scenario in {"mixed", "linear", "break"} else "none",
                "true_lag": 1 if scenario in {"mixed", "linear", "break"} else 0,
            },
            {
                "variable": "x_nonlinear",
                "true_precedence": scenario in {"mixed", "nonlinear"},
                "functional_form": "quadratic" if scenario in {"mixed", "nonlinear"} else "none",
                "true_lag": 1 if scenario in {"mixed", "nonlinear"} else 0,
            },
            {"variable": "x_null", "true_precedence": False, "functional_form": "none", "true_lag": 0},
            {"variable": "x_common", "true_precedence": False, "functional_form": "common driver only", "true_lag": 0},
        ]
    )
    return df, truth


def _leave_one_out_time_mean(values: pd.Series, times: pd.Series) -> pd.Series:
    """Return a time-specific mean that excludes the current unit's value."""
    frame = pd.DataFrame({"value": values.astype(float), "time": times})
    grouped = frame.groupby("time", observed=True)["value"]
    total = grouped.transform("sum")
    count = grouped.transform("count")
    valid = frame["value"].notna()
    numerator = total - frame["value"].where(valid, 0.0)
    denominator = count - valid.astype(int)
    out = numerator / denominator.where(denominator > 0)
    return out.astype(float)


def add_lags(df: pd.DataFrame, variables: Iterable[str], max_lag: int = 2) -> pd.DataFrame:
    out = df.sort_values(["unit", "time"]).copy()
    for v in variables:
        for lag in range(1, max_lag + 1):
            out[f"{v}_lag{lag}"] = out.groupby("unit", observed=True)[v].shift(lag)
    # CCE-style proxies: lagged, leave-one-unit-out cross-sectional means.
    for lag in range(1, max_lag + 1):
        for v in ["y", *[x for x in variables if x != "y"]]:
            col = f"{v}_lag{lag}"
            out[f"mean_{col}"] = _leave_one_out_time_mean(out[col], out["time"])
    return out


def _fit_formula(data: pd.DataFrame, formula: str, cluster: bool = True):
    model = smf.ols(formula=formula, data=data).fit()
    if cluster:
        model = model.get_robustcov_results(cov_type="cluster", groups=data["unit"])
    return model


def classical_and_bayes(
    df: pd.DataFrame,
    variables: list[str],
    max_lag: int = 2,
    two_way_fe: bool = True,
    cce: bool = False,
) -> pd.DataFrame:
    if two_way_fe and cce:
        raise ValueError(
            "Use full time effects or lagged cross-sectional averages as alternative "
            "common-shock controls; do not enable both without a rank-identified design."
        )
    lagged = add_lags(df, ["y", *variables], max_lag=max_lag)
    records: list[dict[str, float | str | int]] = []

    for variable in variables:
        pvals: list[float] = []
        bfs: list[float] = []
        lags: list[int] = []
        for p in range(1, max_lag + 1):
            needed = [f"y_lag{i}" for i in range(1, p + 1)] + [f"{variable}_lag{i}" for i in range(1, p + 1)]
            if cce:
                needed += [f"mean_y_lag{i}" for i in range(1, p + 1)]
                needed += [f"mean_{variable}_lag{i}" for i in range(1, p + 1)]
            dat = lagged.dropna(subset=["y", *needed]).copy()
            y_terms = [f"y_lag{i}" for i in range(1, p + 1)]
            x_terms = [f"{variable}_lag{i}" for i in range(1, p + 1)]
            base_terms = y_terms + ["C(unit)"]
            if two_way_fe:
                base_terms.append("C(time)")
            if cce:
                base_terms.extend([f"mean_y_lag{i}" for i in range(1, p + 1)])
                base_terms.extend([f"mean_{variable}_lag{i}" for i in range(1, p + 1)])
            f0 = "y ~ " + " + ".join(base_terms)
            f1 = f0 + " + " + " + ".join(x_terms)

            m0_plain = smf.ols(f0, dat).fit()
            m1_plain = smf.ols(f1, dat).fit()
            m1 = m1_plain.get_robustcov_results(cov_type="cluster", groups=dat["unit"])
            # Cluster-robust Wald test for the joint null that all x lags are zero.
            names = list(m1_plain.params.index)
            R = np.zeros((len(x_terms), len(names)))
            for r, term in enumerate(x_terms):
                R[r, names.index(term)] = 1.0
            try:
                pval = float(m1.wald_test(R, use_f=True, scalar=True).pvalue)
            except Exception:
                pval = 1.0
            # BIC approximation to BF10, computed on identical estimation samples.
            log_bf = 0.5 * (m0_plain.bic - m1_plain.bic)
            bf = float(np.exp(np.clip(log_bf, -40, 40)))
            pvals.append(pval)
            bfs.append(bf)
            lags.append(p)

        finite = np.isfinite(pvals)
        if not any(finite):
            best_idx = 0
            p_min = 1.0
            n_tested = 0
        else:
            best_idx = int(np.nanargmin(pvals))
            p_min = float(pvals[best_idx])
            n_tested = int(np.sum(finite))
        p_bonf = min(1.0, p_min * max(1, n_tested))
        bf_best_idx = int(np.nanargmax(bfs))
        records.append(
            {
                "variable": variable,
                "best_lag_classical": int(lags[best_idx]),
                "p_raw_min": p_min,
                "p_lag_bonf": p_bonf,
                "best_lag_bayes": int(lags[bf_best_idx]),
                "bf10_bic_max": float(bfs[bf_best_idx]),
            }
        )

    out = pd.DataFrame(records)
    out["q_bh"] = multipletests(out["p_lag_bonf"].to_numpy(), method="fdr_bh")[1]
    return out


def _blocked_splits(times: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.array(sorted(np.unique(times)))
    n = len(unique)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for frac in (0.50, 0.65, 0.80):
        cut = max(5, int(math.floor(n * frac)))
        if cut >= n - 1:
            continue
        width = max(2, min(4, n - cut))
        train_t = unique[:cut]
        test_t = unique[cut : cut + width]
        if len(test_t):
            splits.append((train_t, test_t))
    return splits


def _rf_cv_delta(
    dat: pd.DataFrame,
    variable: str,
    max_lag: int,
    seed: int,
    n_estimators: int,
) -> tuple[float, float]:
    y_terms = [f"y_lag{i}" for i in range(1, max_lag + 1)]
    x_terms = [f"{variable}_lag{i}" for i in range(1, max_lag + 1)]
    cce_terms = [f"mean_y_lag{i}" for i in range(1, max_lag + 1)] + [
        f"mean_{variable}_lag{i}" for i in range(1, max_lag + 1)
    ]
    base = dat[["unit", "time", "y", *y_terms, *x_terms, *cce_terms]].dropna().copy()
    unit_dummies = pd.get_dummies(base["unit"], prefix="unit", drop_first=True, dtype=float)
    base = pd.concat([base.reset_index(drop=True), unit_dummies.reset_index(drop=True)], axis=1)
    unit_terms = list(unit_dummies.columns)
    base["time_scaled"] = (base["time"] - base["time"].min()) / max(base["time"].max() - base["time"].min(), 1)
    restricted = [*y_terms, *cce_terms, "time_scaled", *unit_terms]
    full = [*restricted, *x_terms]

    deltas: list[float] = []
    for fold, (train_t, test_t) in enumerate(_blocked_splits(base["time"].to_numpy())):
        tr = base[base["time"].isin(train_t)]
        te = base[base["time"].isin(test_t)]
        if len(tr) < 80 or len(te) < 20:
            continue
        params = dict(
            n_estimators=n_estimators,
            max_depth=6,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=seed + fold,
            n_jobs=-1,
        )
        r0 = RandomForestRegressor(**params).fit(tr[restricted], tr["y"])
        r1 = RandomForestRegressor(**params).fit(tr[full], tr["y"])
        mse0 = mean_squared_error(te["y"], r0.predict(te[restricted]))
        mse1 = mean_squared_error(te["y"], r1.predict(te[full]))
        delta = 100.0 * (mse0 - mse1) / max(mse0, 1e-12)
        deltas.append(float(delta))
    if not deltas:
        return math.nan, math.nan
    return float(np.median(deltas)), float(np.mean(np.array(deltas) > 0))


def _circular_shift_raw_predictor(
    df: pd.DataFrame, variable: str, seed: int
) -> pd.DataFrame:
    """Shift the raw predictor once per unit, then rebuild all lags."""
    rng = np.random.default_rng(seed)
    out = df.sort_values(["unit", "time"]).copy()
    for _, idx in out.groupby("unit", sort=False, observed=True).groups.items():
        idx = list(idx)
        vals = out.loc[idx, variable].to_numpy().copy()
        if len(vals) > 1:
            k = int(rng.integers(1, len(vals)))
            out.loc[idx, variable] = np.roll(vals, k)
    return out


def rf_evidence(
    df: pd.DataFrame,
    variables: list[str],
    max_lag: int = 2,
    seed: int = 20260723,
    n_estimators: int = 250,
    n_shadow: int = 20,
) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for j, variable in enumerate(variables):
        lagged = add_lags(df, ["y", variable], max_lag=max_lag)
        delta, fold_share = _rf_cv_delta(
            lagged, variable, max_lag, seed + 1000 * j, n_estimators
        )
        shadow = []
        for rep in range(n_shadow):
            shifted_raw = _circular_shift_raw_predictor(
                df, variable, seed + 1000 * j + 100 + rep
            )
            shifted_lagged = add_lags(
                shifted_raw, ["y", variable], max_lag=max_lag
            )
            d, _ = _rf_cv_delta(
                shifted_lagged,
                variable,
                max_lag,
                seed + 1000 * j + 100 + rep,
                max(80, n_estimators // 2),
            )
            if np.isfinite(d):
                shadow.append(d)
        q95 = float(np.quantile(shadow, 0.95)) if shadow else math.nan
        records.append(
            {
                "variable": variable,
                "rf_delta_mse_pct": delta,
                "rf_positive_fold_share": fold_share,
                "rf_shadow_q95": q95,
                "rf_detect": bool(
                    np.isfinite(delta)
                    and np.isfinite(q95)
                    and delta > max(0.0, q95)
                    and fold_share >= 2 / 3
                ),
            }
        )
    return pd.DataFrame(records)


def classify(row: pd.Series) -> str:
    votes = int(row["classical_detect"]) + int(row["bayes_detect"]) + int(row["rf_detect"])
    if votes >= 2:
        return "triangulated"
    if votes == 1 and bool(row["rf_detect"]):
        return "nonlinear-only; replicate before interpretation"
    if votes == 1:
        return "single-family evidence; provisional"
    return "no supported precedence"


def run_demo(out_dir: Path, seed: int, rf_trees: int = 120, rf_shadows: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)
    outputs = out_dir / "outputs"
    outputs.mkdir(exist_ok=True)

    demo, truth = simulate_panel("mixed", n_units=30, n_time=28, seed=seed)
    null, null_truth = simulate_panel("null", n_units=30, n_time=28, seed=seed + 1)
    common, common_truth = simulate_panel("common_shock", n_units=30, n_time=28, seed=seed + 2)
    break_df, break_truth = simulate_panel("break", n_units=30, n_time=28, seed=seed + 3)

    demo.to_csv(data_dir / "synthetic_mixed_panel.csv", index=False)
    null.to_csv(data_dir / "synthetic_null_panel.csv", index=False)
    common.to_csv(data_dir / "synthetic_common_driver_panel.csv", index=False)
    break_df.to_csv(data_dir / "synthetic_structural_break_panel.csv", index=False)
    truth.assign(dataset="synthetic_mixed_panel.csv").to_csv(data_dir / "truth_mixed.csv", index=False)
    pd.concat(
        [
            truth.assign(dataset="synthetic_mixed_panel.csv"),
            null_truth.assign(dataset="synthetic_null_panel.csv"),
            common_truth.assign(dataset="synthetic_common_driver_panel.csv"),
            break_truth.assign(dataset="synthetic_structural_break_panel.csv"),
        ],
        ignore_index=True,
    ).to_csv(data_dir / "truth_all.csv", index=False)

    variables = ["x_linear", "x_nonlinear", "x_null", "x_common"]
    cb = classical_and_bayes(demo, variables, max_lag=2, two_way_fe=True, cce=False)
    rf = rf_evidence(demo, variables, max_lag=2, seed=seed, n_estimators=rf_trees, n_shadow=rf_shadows)
    evidence = truth.merge(cb, on="variable").merge(rf, on="variable")
    evidence["classical_detect"] = evidence["q_bh"] < 0.05
    evidence["bayes_detect"] = evidence["bf10_bic_max"] >= 5.0
    evidence["classification"] = evidence.apply(classify, axis=1)
    evidence.to_csv(outputs / "demo_evidence.csv", index=False)

    # Sensitivity to common-driver adjustment: x_common is not a true cause.
    common_unadj = classical_and_bayes(
        common, ["x_common"], max_lag=2, two_way_fe=False, cce=False
    )
    common_time = classical_and_bayes(
        common, ["x_common"], max_lag=2, two_way_fe=True, cce=False
    )
    common_cce = classical_and_bayes(
        common, ["x_common"], max_lag=2, two_way_fe=False, cce=True
    )
    common_sens = pd.concat(
        [
            common_unadj.assign(specification="unit fixed effects only"),
            common_time.assign(specification="unit and time fixed effects"),
            common_cce.assign(
                specification="unit effects plus lagged cross-sectional averages"
            ),
        ],
        ignore_index=True,
    )
    common_sens.to_csv(outputs / "common_driver_sensitivity.csv", index=False)

    # The 100-replication sensitivity matrix is generated separately by
    # run_sensitivity_matrix.py so that this demo generator remains quick and
    # does not overwrite the fixed sensitivity outputs.

    manifest = {
        "seed": seed,
        "n_units_demo": 30,
        "n_time_demo": 28,
        "max_lag": 2,
        "lag_search_alpha": 0.05,
        "fdr_q": 0.05,
        "bayes_factor_threshold": 5,
        "rf_n_estimators_demo": rf_trees,
        "rf_shadow_repetitions_demo": rf_shadows,
        "common_factor_control": "Full time effects are the primary demonstration; lagged cross-sectional averages are an alternative sensitivity specification.",
        "shadow_construction": "Non-zero circular shift of the raw predictor within unit, followed by lag reconstruction.",
        "sensitivity_matrix_command": "python python/run_sensitivity_matrix.py --out outputs --replications 100",
        "warning": "Synthetic outputs are method-validation smoke tests, not universal power or size guarantees and not estimates from the trade application.",
    }
    (outputs / "validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate synthetic panels and fixed demo evidence.")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--rf-trees", type=int, default=120)
    parser.add_argument("--rf-shadows", type=int, default=8)
    parser.add_argument("--quick", action="store_true", help="Use a smaller Random Forest for a rapid smoke run.")
    args = parser.parse_args()
    trees = 60 if args.quick else args.rf_trees
    shadows = 3 if args.quick else args.rf_shadows
    run_demo(args.out, args.seed, rf_trees=trees, rf_shadows=shadows)
