from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as fdist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from generate_validation_data import simulate_panel  # noqa: E402


def two_way_within(values: np.ndarray, units: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Remove unit and time means from one balanced-panel variable."""
    series = pd.Series(values, dtype=float)
    unit_mean = series.groupby(units).transform("mean").to_numpy()
    time_mean = series.groupby(times).transform("mean").to_numpy()
    return np.asarray(values, dtype=float) - unit_mean - time_mean + float(np.nanmean(values))


def ols_stats(y: np.ndarray, x: np.ndarray) -> tuple[float, float, int, int]:
    """Return RSS, BIC, numerical rank, and n for a stable least-squares fit."""
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    rss = float(residual @ residual)
    n = len(y)
    k = int(rank)
    bic = n * math.log(max(rss / n, 1e-12)) + k * math.log(n)
    return rss, bic, k, n


def test_fast(df: pd.DataFrame, variable: str, max_lag: int = 2) -> tuple[float, float]:
    """Fast two-way-within lag test used only for the simulation smoke matrix."""
    data = df.sort_values(["unit", "time"]).copy()
    for lag in range(1, max_lag + 1):
        data[f"y_lag{lag}"] = data.groupby("unit", observed=True)["y"].shift(lag)
        data[f"x_lag{lag}"] = data.groupby("unit", observed=True)[variable].shift(lag)

    p_values: list[float] = []
    bayes_factors: list[float] = []
    for p in range(1, max_lag + 1):
        columns = ["y", *[f"y_lag{i}" for i in range(1, p + 1)], *[f"x_lag{i}" for i in range(1, p + 1)]]
        z = data.dropna(subset=columns).copy()
        unit = z["unit"].to_numpy()
        time = z["time"].to_numpy()
        y = two_way_within(z["y"].to_numpy(), unit, time)
        y_lags = np.column_stack(
            [two_way_within(z[f"y_lag{i}"].to_numpy(), unit, time) for i in range(1, p + 1)]
        )
        x_lags = np.column_stack(
            [two_way_within(z[f"x_lag{i}"].to_numpy(), unit, time) for i in range(1, p + 1)]
        )
        x0 = np.column_stack([np.ones(len(z)), y_lags])
        x1 = np.column_stack([x0, x_lags])
        rss0, bic0, k0, n = ols_stats(y, x0)
        rss1, bic1, k1, _ = ols_stats(y, x1)
        q = max(k1 - k0, 1)
        df2 = max(n - k1, 1)
        f_stat = max(((rss0 - rss1) / q) / max(rss1 / df2, 1e-12), 0.0)
        p_values.append(float(fdist.sf(f_stat, q, df2)))
        bayes_factors.append(float(np.exp(np.clip(0.5 * (bic0 - bic1), -40, 40))))

    lag_adjusted_p = min(1.0, min(p_values) * len(p_values))
    return lag_adjusted_p, max(bayes_factors)


def run_matrix(out_dir: Path, seed: int, replications: int) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    scenarios = [
        ("null", "x_linear"),
        ("linear", "x_linear"),
        ("nonlinear", "x_nonlinear"),
        ("break", "x_linear"),
    ]
    for n_time in (20, 28, 40):
        for scenario_index, (scenario, target) in enumerate(scenarios):
            for rep in range(replications):
                panel, _ = simulate_panel(
                    scenario,
                    n_units=20,
                    n_time=n_time,
                    seed=seed + 100000 + n_time * 1000 + rep + scenario_index * 10000,
                )
                p_value, bf10 = test_fast(panel, target, max_lag=2)
                records.append(
                    {
                        "scenario": scenario,
                        "n_time": n_time,
                        "rep": rep + 1,
                        "method": "classical_two_way_within",
                        "detected": p_value < 0.05,
                    }
                )
                records.append(
                    {
                        "scenario": scenario,
                        "n_time": n_time,
                        "rep": rep + 1,
                        "method": "bayesian_bic",
                        "detected": bf10 >= 5.0,
                    }
                )

    replications_table = pd.DataFrame(records)
    replications_table.to_csv(out_dir / "simulation_replications.csv", index=False)
    summary = (
        replications_table.groupby(["scenario", "n_time", "method"], as_index=False)
        .agg(detection_rate=("detected", "mean"), n_rep=("detected", "size"))
        .sort_values(["scenario", "n_time", "method"])
    )
    summary.to_csv(out_dir / "simulation_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate the short-panel sensitivity matrix.")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--replications", type=int, default=100)
    args = parser.parse_args()
    result = run_matrix(args.out, args.seed, args.replications)
    print(result.to_string(index=False))
