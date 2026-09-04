from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from r1_core import Cell, simulate_panel, stable_seed
from r1_rf_core import rf_edge


def one_rep(payload: tuple[dict[str, Any], dict[str, Any], int]) -> list[dict[str, Any]]:
    base_cfg, tcfg, rep = payload
    r = tcfg["rf_fidelity"]
    cell = Cell(
        cell_id="rf_reference_fidelity", scenario="partial_null", reps=1,
        T=int(r["T"]), N=int(r["N"]), K=int(r["K"]),
        phi_x=float(r["phi_x"]), phi_y=float(r["phi_y"]),
        cross_dependence=float(r["cross_dependence"]),
        linear_effect=0.0, nonlinear_effect=float(r["nonlinear_effect"]),
        n_true_linear=0, n_true_nonlinear=1,
        correlated_predictor_blocks=True, family="rf_reference_fidelity",
    )
    seed = stable_seed(int(tcfg["seed"]), "rf-reference-fidelity", rep)
    Y, X, truth = simulate_panel(cell, seed)
    predictors = [0, cell.K - 1]
    rows: list[dict[str, Any]] = []
    for k in predictors:
        for setting in r["settings"]:
            for seed_index in range(1, int(r["seed_repeats"]) + 1):
                ans = rf_edge(
                    Y, X[:, :, k], base_cfg,
                    stable_seed(seed, k, setting["name"], seed_index),
                    trees=int(setting["trees"]),
                    shadow_repeats=int(setting["shadows"]),
                )
                threshold = float(r["fold_share_threshold"])
                supported = bool(
                    np.isfinite(ans["rf_mean"]) and ans["rf_mean"] > 0
                    and np.isfinite(ans["rf_shadow_q95"])
                    and ans["rf_mean"] > ans["rf_shadow_q95"]
                    and ans["rf_fold_share"] >= threshold
                )
                rows.append({
                    "rep": rep, "predictor": k + 1, "truth": bool(truth[k]),
                    "setting": str(setting["name"]), "trees": int(setting["trees"]),
                    "shadows": int(setting["shadows"]), "seed_index": seed_index,
                    "supported": supported,
                    "rf_mean": float(ans["rf_mean"]),
                    "rf_shadow_q95": float(ans["rf_shadow_q95"]),
                    "rf_fold_share": float(ans["rf_fold_share"]),
                    "rf_emp_p": float(ans["rf_emp_p"]),
                })
    return rows


def kappa_binary(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return np.nan
    a = a.astype(bool)
    b = b.astype(bool)
    po = float(np.mean(a == b))
    pa = float(np.mean(a))
    pb = float(np.mean(b))
    pe = pa * pb + (1.0 - pa) * (1.0 - pb)
    return float((po - pe) / (1.0 - pe)) if abs(1.0 - pe) > 1e-12 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description="RF Monte Carlo approximation fidelity check")
    ap.add_argument("--config", default="config/r1_targeted_checks.yml")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tcfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load(Path(tcfg["base_config"]).read_text(encoding="utf-8"))
    out = Path(args.out or tcfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    chunk_dir = out / "rf_fidelity_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    reps = int(tcfg["rf_fidelity"]["replications"])

    missing = [r for r in range(1, reps + 1) if not (chunk_dir / f"rep_{r:04d}.csv").exists()]
    payloads = [(base_cfg, tcfg, r) for r in missing]
    workers = max(1, int(args.workers))
    if workers == 1:
        for j, p in enumerate(payloads, 1):
            pd.DataFrame(one_rep(p)).to_csv(chunk_dir / f"rep_{p[2]:04d}.csv", index=False)
            print(f"RF fidelity: {j}/{len(payloads)} new reps", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one_rep, p): p[2] for p in payloads}
            for j, fut in enumerate(as_completed(futs), 1):
                rep = futs[fut]
                pd.DataFrame(fut.result()).to_csv(chunk_dir / f"rep_{rep:04d}.csv", index=False)
                print(f"RF fidelity: {j}/{len(payloads)} new reps", flush=True)

    raw = pd.concat([pd.read_csv(f) for f in sorted(chunk_dir.glob("rep_*.csv"))], ignore_index=True)
    raw.to_csv(out / "rf_fidelity_raw_R1.csv", index=False)

    summary = raw.groupby(["truth", "setting", "trees", "shadows"], as_index=False).agg(
        decision_rate=("supported", "mean"),
        mean_gain=("rf_mean", "mean"),
        mean_shadow_q95=("rf_shadow_q95", "mean"),
        mean_fold_share=("rf_fold_share", "mean"),
        n=("supported", "size"),
    )
    summary.to_csv(out / "rf_fidelity_summary_R1.csv", index=False)

    ds = raw.groupby(["rep", "predictor", "truth", "setting"], as_index=False).agg(
        seed_support_share=("supported", "mean"),
        gain_mean=("rf_mean", "mean"),
        gain_sd=("rf_mean", "std"),
        q95_mean=("rf_shadow_q95", "mean"),
    )
    ds["seed_disagreement"] = (ds["seed_support_share"] > 0) & (ds["seed_support_share"] < 1)
    ds.to_csv(out / "rf_fidelity_seed_variability_R1.csv", index=False)

    ref = str(tcfg["rf_fidelity"]["reference_setting"])
    settings = [str(s["name"]) for s in tcfg["rf_fidelity"]["settings"]]
    agreement_rows: list[dict[str, Any]] = []
    for setting in settings:
        if setting == ref:
            continue
        for truth_filter in [None, False, True]:
            a = ds[ds["setting"].eq(setting)].copy()
            b = ds[ds["setting"].eq(ref)].copy()
            keys = ["rep", "predictor", "truth"]
            z = a.merge(b, on=keys, suffixes=("_a", "_b"))
            if truth_filter is not None:
                z = z[z["truth"].astype(bool).eq(bool(truth_filter))]
            ma = z["seed_support_share_a"].to_numpy() >= 0.5
            mb = z["seed_support_share_b"].to_numpy() >= 0.5
            agreement_rows.append({
                "setting": setting, "reference": ref,
                "truth": "all" if truth_filter is None else str(bool(truth_filter)),
                "n_datasets": int(len(z)),
                "majority_decision_agreement": float(np.mean(ma == mb)) if len(z) else np.nan,
                "cohen_kappa": kappa_binary(ma, mb),
                "decision_rate_setting": float(np.mean(ma)) if len(z) else np.nan,
                "decision_rate_reference": float(np.mean(mb)) if len(z) else np.nan,
                "decision_rate_abs_diff": float(abs(np.mean(ma) - np.mean(mb))) if len(z) else np.nan,
                "mean_gain_correlation": float(np.corrcoef(z["gain_mean_a"], z["gain_mean_b"])[0, 1]) if len(z) > 1 else np.nan,
            })
    pd.DataFrame(agreement_rows).to_csv(out / "rf_fidelity_setting_agreement_R1.csv", index=False)
    print("RF reference-fidelity check completed.", flush=True)


if __name__ == "__main__":
    main()
