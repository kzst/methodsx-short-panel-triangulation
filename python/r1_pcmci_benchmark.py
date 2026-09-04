from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests


def _one_rep(payload: tuple[pd.DataFrame, pd.DataFrame, list[str], str, int, int, float, float]) -> list[dict[str, Any]]:
    z, truth_rep, xcols, scenario, T, tau_max, pc_alpha, q_level = payload
    import tigramite.data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    rep = int(z["rep"].iloc[0])
    var_names = ["y", *xcols]
    datasets: dict[int, np.ndarray] = {}
    for unit, du in z.groupby("unit", sort=True):
        du = du.sort_values("time")
        datasets[int(unit)] = du[var_names].to_numpy(dtype=float)
    frame = pp.DataFrame(datasets, analysis_mode="multiple", var_names=var_names)
    pcmci = PCMCI(dataframe=frame, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    result = pcmci.run_pcmci(
        tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha, alpha_level=q_level
    )
    pmat = np.asarray(result["p_matrix"], dtype=float)
    p_edge = np.ones(len(xcols), dtype=float)
    best_lag = np.ones(len(xcols), dtype=int)
    for k in range(len(xcols)):
        vals = pmat[k + 1, 0, 1:tau_max + 1]
        finite = np.isfinite(vals)
        if np.any(finite):
            j = int(np.nanargmin(vals))
            p_edge[k] = min(1.0, float(vals[j]) * int(np.sum(finite)))
            best_lag[k] = j + 1
    q_by = multipletests(p_edge, method="fdr_by")[1]
    tr = truth_rep.sort_values("predictor")
    rows = []
    for k in range(len(xcols)):
        rows.append({
            "scenario": scenario, "T": T, "rep": rep, "predictor": k + 1,
            "truth": bool(tr.iloc[k]["truth"]), "signal_type": str(tr.iloc[k]["signal_type"]),
            "pcmci_p_lag": float(p_edge[k]), "pcmci_q_by": float(q_by[k]),
            "pcmci_best_lag": int(best_lag[k]), "pcmci_supported": bool(q_by[k] <= q_level),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="PCMCI benchmark for MethodsX R1 validation")
    ap.add_argument("--config", default="config/r1_validation_full.yml")
    ap.add_argument("--out", default="outputs/R1")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    inputs = out / "benchmark_inputs"
    if not inputs.exists():
        raise SystemExit(f"Benchmark input directory not found: {inputs}")

    try:
        import tigramite  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Tigramite is required for the reviewer-requested PCMCI benchmark. "
            "Install it with: python3 -m pip install tigramite\n" + str(exc)
        )

    tau_max = int(cfg["benchmarks"]["pcmci_tau_max"])
    pc_alpha = float(cfg["benchmarks"]["pcmci_pc_alpha"])
    q_level = float(cfg["fdr_q"])
    workers = max(1, int(args.workers))
    records: list[dict[str, Any]] = []

    files = sorted(inputs.glob("benchmark_*_T*.csv.gz"))
    files = [f for f in files if "_truth" not in f.name]
    for f in files:
        stem = f.name.replace(".csv.gz", "")
        T = int(stem.split("_T")[-1])
        scenario = stem.replace("benchmark_", "").rsplit("_T", 1)[0]
        truth_file = inputs / f"{stem}_truth.csv"
        if not truth_file.exists():
            continue
        dat = pd.read_csv(f)
        truth = pd.read_csv(truth_file)
        xcols = sorted(
            [c for c in dat.columns if c.startswith("x") and c[1:].isdigit()],
            key=lambda z: int(z[1:]),
        )
        payloads = []
        for rep in sorted(dat["rep"].unique()):
            payloads.append((
                dat.loc[dat["rep"] == rep].copy(), truth.loc[truth["rep"] == rep].copy(),
                xcols, scenario, T, tau_max, pc_alpha, q_level,
            ))
        if workers <= 1:
            for j, p in enumerate(payloads, 1):
                records.extend(_one_rep(p))
                if j % 25 == 0:
                    print(f"PCMCI {scenario} T={T}: {j}/{len(payloads)}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_one_rep, p) for p in payloads]
                for j, fut in enumerate(as_completed(futs), 1):
                    records.extend(fut.result())
                    if j % 25 == 0:
                        print(f"PCMCI {scenario} T={T}: {j}/{len(payloads)}", flush=True)

    res = pd.DataFrame(records)
    res.to_csv(out / "benchmark_pcmci_R1.csv", index=False)
    summary_rows = []
    for (scenario, T), g in res.groupby(["scenario", "T"]):
        d = g["pcmci_supported"].astype(bool)
        null = ~g["truth"].astype(bool)
        alt = g["truth"].astype(bool)
        fdp = []
        for _, gr in g.groupby("rep"):
            ds = gr["pcmci_supported"].astype(bool)
            nsel = int(ds.sum())
            fdp.append(float(np.sum(ds & ~gr["truth"].astype(bool)) / nsel) if nsel else 0.0)
        summary_rows.append({
            "method": "PCMCI-ParCorr", "scenario": scenario, "T": int(T),
            "n_rep": int(g["rep"].nunique()),
            "false_positive_rate": float(d[null].mean()) if int(null.sum()) else math.nan,
            "power_all": float(d[alt].mean()) if int(alt.sum()) else math.nan,
            "power_linear": float(d[g["signal_type"].eq("linear")].mean()) if g["signal_type"].eq("linear").any() else math.nan,
            "power_nonlinear": float(d[g["signal_type"].eq("nonlinear")].mean()) if g["signal_type"].eq("nonlinear").any() else math.nan,
            "empirical_fdr": float(np.mean(fdp)),
        })
    pd.DataFrame(summary_rows).to_csv(out / "benchmark_pcmci_summary_R1.csv", index=False)
    print("R1 PCMCI benchmark stage completed.", flush=True)


if __name__ == "__main__":
    main()
