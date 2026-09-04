from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def rate(x: pd.Series) -> float:
    return float(x.astype(float).mean()) if len(x) else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-process completed R1 simulation without rerunning Monte Carlo")
    ap.add_argument("--config", default="config/r1_targeted_checks.yml")
    ap.add_argument("--source", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source = Path(args.source or cfg["source_output"])
    out = Path(args.out or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    cell_dirs = sorted((source / "simulation_chunks").glob("*"))
    if not cell_dirs:
        raise SystemExit(f"No simulation chunks found below {source}")

    summary_rows: list[dict] = []
    tier_rows: list[dict] = []
    alpha_levels = [float(x) for x in cfg["alpha_levels"]]
    q_level = float(cfg["fdr_q"])

    for cdir in cell_dirs:
        files = sorted(cdir.glob("edges_*.csv.gz"))
        if not files:
            continue
        g = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
        first = g.iloc[0]
        base = {
            "cell_id": str(first["cell_id"]),
            "family": str(first["family"]),
            "scenario": str(first["scenario"]),
            "T": int(first["T"]),
            "N": int(first["N"]),
            "K": int(first["K"]),
            "n_edge_rep": int(len(g)),
        }
        truth = g["truth"].astype(bool)
        null = ~truth
        linear = g["signal_type"].eq("linear")
        nonlinear = g["signal_type"].eq("nonlinear")

        row = dict(base)
        for a in alpha_levels:
            dec = g["p_lag"].astype(float) <= a
            tag = str(a).replace("0.", "").replace(".", "p")
            row[f"raw_alpha_{tag}_false_positive_rate"] = rate(dec[null])
            row[f"raw_alpha_{tag}_power_all"] = rate(dec[truth])
            row[f"raw_alpha_{tag}_power_linear"] = rate(dec[linear])
            row[f"raw_alpha_{tag}_power_nonlinear"] = rate(dec[nonlinear])
        for label, col in [("bh", "q_bh"), ("by", "q_by")]:
            dec = g[col].astype(float) <= q_level
            row[f"{label}_q05_false_positive_rate"] = rate(dec[null])
            row[f"{label}_q05_power_all"] = rate(dec[truth])
            row[f"{label}_q05_power_linear"] = rate(dec[linear])
            row[f"{label}_q05_power_nonlinear"] = rate(dec[nonlinear])
        summary_rows.append(row)

        for rule_name, col in [("submitted", "tier_submitted"), ("grouped", "tier_grouped")]:
            for signal_name, mask in [("null", null), ("linear", linear), ("nonlinear", nonlinear)]:
                z = g.loc[mask, col].astype(str)
                if z.empty:
                    continue
                vc = z.value_counts(dropna=False)
                for tier, n in vc.items():
                    tier_rows.append({
                        **base,
                        "rule": rule_name,
                        "signal_type": signal_name,
                        "tier": tier,
                        "count": int(n),
                        "denominator": int(len(z)),
                        "probability": float(n / len(z)),
                    })

    pd.DataFrame(summary_rows).sort_values(["family", "T", "N", "K"]).to_csv(
        out / "alpha_q_sensitivity_R1.csv", index=False
    )
    pd.DataFrame(tier_rows).sort_values(
        ["family", "T", "rule", "signal_type", "tier"]
    ).to_csv(out / "tier_probability_all_R1.csv", index=False)
    print("Targeted post-processing of completed R1 simulation finished.", flush=True)


if __name__ == "__main__":
    main()
