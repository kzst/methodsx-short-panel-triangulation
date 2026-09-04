from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from r1_core import Cell, simulate_panel, stable_seed


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate assumption-compatible benchmark control panels")
    ap.add_argument("--config", default="config/r1_targeted_checks.yml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    b = cfg["benchmark_control"]
    out = Path(args.out or cfg["output_dir"])
    bdir = out / "benchmark_inputs"
    bdir.mkdir(parents=True, exist_ok=True)
    reps = int(b["replications"])
    K = int(b["K"])

    for T in b["T"]:
        for scenario in ["global_null", "partial_null"]:
            target = bdir / f"benchmark_{scenario}_T{int(T)}.csv.gz"
            truth_target = bdir / f"benchmark_{scenario}_T{int(T)}_truth.csv"
            if target.exists() and truth_target.exists():
                print(f"SKIP existing benchmark input {scenario} T={T}", flush=True)
                continue
            nlin = 0 if scenario == "global_null" else int(b["n_true_linear"])
            nnon = 0 if scenario == "global_null" else int(b["n_true_nonlinear"])
            cell = Cell(
                cell_id=f"target_benchmark_{scenario}_T{int(T)}",
                scenario=scenario,
                reps=reps,
                T=int(T), N=int(b["N"]), K=K,
                phi_x=float(b["phi_x"]), phi_y=float(b["phi_y"]),
                cross_dependence=float(b["cross_dependence"]),
                linear_effect=float(b["linear_effect"]),
                nonlinear_effect=float(b["nonlinear_effect"]),
                n_true_linear=nlin, n_true_nonlinear=nnon,
                correlated_predictor_blocks=bool(b["correlated_predictor_blocks"]),
                family="benchmark_control",
            )
            rows: list[dict] = []
            truth_rows: list[dict] = []
            for rep in range(1, reps + 1):
                seed = stable_seed(int(cfg["seed"]), "benchmark-control", scenario, int(T), rep)
                Y, X, truth = simulate_panel(cell, seed)
                for t in range(cell.T):
                    for i in range(cell.N):
                        row = {"rep": rep, "unit": i + 1, "time": t + 1, "y": float(Y[t, i])}
                        for k in range(K):
                            row[f"x{k+1}"] = float(X[t, i, k])
                        rows.append(row)
                for k in range(K):
                    if truth[k] and k < nlin:
                        st = "linear"
                    elif truth[k] and k < nlin + nnon:
                        st = "nonlinear"
                    else:
                        st = "null"
                    truth_rows.append({
                        "rep": rep, "predictor": k + 1,
                        "truth": bool(truth[k]), "signal_type": st,
                    })
                if rep % 50 == 0:
                    print(f"target benchmark {scenario} T={T}: {rep}/{reps}", flush=True)
            pd.DataFrame(rows).to_csv(target, index=False, compression="gzip")
            pd.DataFrame(truth_rows).to_csv(truth_target, index=False)

    meta = {
        "purpose": "assumption-compatible benchmark control",
        "cross_dependence": float(b["cross_dependence"]),
        "replications_per_T_scenario": reps,
        "T": [int(x) for x in b["T"]],
        "N": int(b["N"]), "K": K,
    }
    (out / "benchmark_control_design_R1.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Targeted benchmark control inputs completed.", flush=True)


if __name__ == "__main__":
    main()
