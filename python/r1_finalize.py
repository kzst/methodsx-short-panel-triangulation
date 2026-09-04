from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate reviewer-facing R1 validation outputs")
    ap.add_argument("--out", default="outputs/R1")
    args = ap.parse_args()
    out = Path(args.out)

    edge_path = out / "branch_and_tier_rates_R1.csv"
    if not edge_path.exists():
        raise SystemExit(f"Missing {edge_path}")
    edge = pd.read_csv(edge_path)

    dep_cols = [c for c in edge.columns if c.startswith("phi_") or c.startswith("joint_") or c.startswith("p_")]
    dep = edge[["cell_id", "family", "scenario", *dep_cols]].copy()
    dep.to_csv(out / "branch_dependence_R1.csv", index=False)

    bench_rows = []
    primary = edge[edge["family"].isin(["primary_global_null", "primary_partial_null"])].copy()
    for _, r in primary.iterrows():
        bench_rows.append({
            "method": "R1 workflow: grouped-family triangulated tier",
            "scenario": r["scenario"],
            "T": r.get("T", np.nan),
            "false_positive_rate": r.get("grouped_triangulated_null_selection_rate", np.nan),
            "power_all": r.get("grouped_triangulated_power", np.nan),
            "empirical_fdr": np.nan,
            "source": "tier simulation",
        })
        bench_rows.append({
            "method": "R1 workflow: classical BY branch",
            "scenario": r["scenario"],
            "T": r.get("T", np.nan),
            "false_positive_rate": r.get("classical_by_false_positive_rate", np.nan),
            "power_all": r.get("classical_by_power", np.nan),
            "empirical_fdr": np.nan,
            "source": "tier simulation",
        })

    for path in [out / "benchmark_panel_granger_summary_R1.csv", out / "benchmark_pcmci_summary_R1.csv"]:
        if path.exists():
            b = pd.read_csv(path)
            for _, r in b.iterrows():
                bench_rows.append({
                    "method": r.get("method", path.stem),
                    "scenario": r.get("scenario", ""),
                    "T": r.get("T", np.nan),
                    "false_positive_rate": r.get("false_positive_rate", np.nan),
                    "power_all": r.get("power_all", np.nan),
                    "empirical_fdr": r.get("empirical_fdr", np.nan),
                    "source": path.name,
                })
    pd.DataFrame(bench_rows).to_csv(out / "reviewer_benchmark_comparison_R1.csv", index=False)

    tuning_path = out / "rf_tuning_stability_R1.csv"
    if tuning_path.exists():
        t = pd.read_csv(tuning_path)
        group = ["truth", "trees", "shadows", "fold_share_threshold"]
        tsum = t.groupby(group, dropna=False).agg(
            decision_rate=("supported_at_threshold", "mean"),
            mean_gain=("rf_mean", "mean"),
            mean_shadow_q95=("rf_shadow_q95", "mean"),
            mean_fold_share=("rf_fold_share", "mean"),
            n=("supported_at_threshold", "size"),
        ).reset_index()
        tsum.to_csv(out / "rf_tuning_stability_summary_R1.csv", index=False)
        seed_group = ["rep", "predictor", "truth", "trees", "shadows", "fold_share_threshold"]
        seed_rates = t.groupby(seed_group, dropna=False)["supported_at_threshold"].mean().reset_index(name="seed_support_share")
        seed_summary = seed_rates.groupby(["truth", "trees", "shadows", "fold_share_threshold"], dropna=False).agg(
            mean_seed_support_share=("seed_support_share", "mean"),
            mean_seed_disagreement=("seed_support_share", lambda x: float(np.mean((x > 0) & (x < 1)))),
            n_datasets=("seed_support_share", "size"),
        ).reset_index()
        seed_summary.to_csv(out / "rf_seed_variability_summary_R1.csv", index=False)

    null_path = out / "null_calibration_R1.csv"
    if null_path.exists():
        z = pd.read_csv(null_path)
        nsum = z.groupby(["T", "block_length"], dropna=False).agg(
            classical_raw_size=("classical_p_lag", lambda x: float(np.mean(np.asarray(x, dtype=float) <= 0.05))),
            rf_circular_false_positive=("rf_circular_supported", "mean"),
            rf_block_false_positive=("rf_block_supported", "mean"),
            mean_rf_circular_p=("rf_circular_p", "mean"),
            mean_rf_block_p=("rf_block_p", "mean"),
            n=("rep", "size"),
        ).reset_index()
        nsum.to_csv(out / "null_calibration_summary_R1.csv", index=False)

    limitation = {
        "generated_regressor_uncertainty_propagated": False,
        "r1_action": "limitation",
        "required_manuscript_change": (
            "Remove the unsupported 'fraction of supported runs' inferential interpretation; "
            "state that first-stage PPML/OLS elasticity uncertainty is not propagated, explain "
            "shared year-specific estimation error and cross-sectional dependence, and retain "
            "PPML/OLS reruns only as estimator sensitivity rather than uncertainty propagation."
        ),
    }
    (out / "generated_regressor_scope_R1.json").write_text(json.dumps(limitation, indent=2), encoding="utf-8")

    print("R1 reviewer-facing aggregation completed.", flush=True)


if __name__ == "__main__":
    main()
