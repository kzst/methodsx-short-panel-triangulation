from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from r1_core import Cell, linear_all, log, simulate_panel, stable_seed
from r1_rf_core import assign_tier, placebo_gate, rf_edge, unit_replication_counts


def run_one_replication(cell: Cell, rep: int, cfg: dict[str, Any], base_seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    seed = stable_seed(base_seed, cell.cell_id, rep)
    Y, X, truth = simulate_panel(cell, seed)
    K = cell.K

    lin = linear_all(Y, X, cfg, control="time_fe")
    cce = linear_all(Y, X, cfg, control="cce") if cell.N > 1 else lin.copy()
    class_support = lin["q_by"].to_numpy() <= float(cfg["fdr_q"])
    class_support_bh = lin["q_bh"].to_numpy() <= float(cfg["fdr_q"])
    bayes_support = lin["logbf_max"].to_numpy() >= float(cfg["bayesian"]["log_bf_threshold"])
    cce_class = cce["q_by"].to_numpy() <= float(cfg["fdr_q"])
    cce_bayes = cce["logbf_max"].to_numpy() >= float(cfg["bayesian"]["log_bf_threshold"])

    rf_rows = [
        rf_edge(Y, X[:, :, k], cfg, stable_seed(seed, "rf", k))
        for k in range(K)
    ]
    rf_support = np.array([bool(r["rf_supported"]) for r in rf_rows], dtype=bool)

    sensitivity = np.where(class_support | bayes_support, cce_class | cce_bayes, rf_support)
    reps = unit_replication_counts(Y, X, lin, rf_rows, cfg, seed)
    plc = placebo_gate(Y, X, lin, cfg, seed)

    edge_rows = []
    for k in range(K):
        if truth[k] and k < cell.n_true_linear:
            signal_type = "linear"
        elif truth[k] and k < cell.n_true_linear + cell.n_true_nonlinear:
            signal_type = "nonlinear"
        else:
            signal_type = "null"
        tier, n_methods = assign_tier(
            bool(class_support[k]), bool(bayes_support[k]), bool(rf_support[k]), int(reps[k]),
            bool(plc["placebo_pass"]), bool(sensitivity[k]), grouped=False,
        )
        gtier, n_families = assign_tier(
            bool(class_support[k]), bool(bayes_support[k]), bool(rf_support[k]), int(reps[k]),
            bool(plc["placebo_pass"]), bool(sensitivity[k]), grouped=True,
        )
        edge_rows.append({
            "cell_id": cell.cell_id,
            "family": cell.family,
            "scenario": cell.scenario,
            "T": cell.T,
            "N": cell.N,
            "K": cell.K,
            "phi_x": cell.phi_x,
            "phi_y": cell.phi_y,
            "cross_dependence": cell.cross_dependence,
            "linear_effect": cell.linear_effect,
            "nonlinear_effect": cell.nonlinear_effect,
            "rep": rep,
            "predictor": k + 1,
            "truth": bool(truth[k]),
            "signal_type": signal_type,
            "p_lag": float(lin.loc[k, "p_lag"]),
            "q_bh": float(lin.loc[k, "q_bh"]),
            "q_by": float(lin.loc[k, "q_by"]),
            "classical_bh": bool(class_support_bh[k]),
            "classical_by": bool(class_support[k]),
            "logbf10": float(lin.loc[k, "logbf_max"]),
            "bayes": bool(bayes_support[k]),
            "rf": bool(rf_support[k]),
            "rf_mean": float(rf_rows[k]["rf_mean"]),
            "rf_fold_share": float(rf_rows[k]["rf_fold_share"]),
            "rf_shadow_q95": float(rf_rows[k]["rf_shadow_q95"]),
            "rf_emp_p": float(rf_rows[k]["rf_emp_p"]),
            "cce_classical_by": bool(cce_class[k]),
            "cce_bayes": bool(cce_bayes[k]),
            "sensitivity_pass": bool(sensitivity[k]),
            "replication_count": int(reps[k]),
            "placebo_pass": bool(plc["placebo_pass"]),
            "tier_submitted": tier,
            "n_methods": int(n_methods),
            "tier_grouped": gtier,
            "n_families": int(n_families),
        })
    edges = pd.DataFrame(edge_rows)

    summary: dict[str, Any] = {
        "cell_id": cell.cell_id,
        "family": cell.family,
        "scenario": cell.scenario,
        "rep": rep,
        "seed": seed,
        "T": cell.T,
        "N": cell.N,
        "K": cell.K,
        "phi_x": cell.phi_x,
        "phi_y": cell.phi_y,
        "cross_dependence": cell.cross_dependence,
        "linear_effect": cell.linear_effect,
        "nonlinear_effect": cell.nonlinear_effect,
        **plc,
    }
    for label, col in [("submitted", "tier_submitted"), ("grouped", "tier_grouped")]:
        for tier_name, short in [
            ("triangulated temporal precedence", "triangulated"),
            ("replicated single-family evidence", "replicated_single"),
        ]:
            sel = edges[col].eq(tier_name).to_numpy()
            nsel = int(sel.sum())
            nfalse = int(np.sum(sel & ~truth))
            ntrue = int(np.sum(sel & truth))
            summary[f"{label}_{short}_n"] = nsel
            summary[f"{label}_{short}_false"] = nfalse
            summary[f"{label}_{short}_true"] = ntrue
            summary[f"{label}_{short}_fdp"] = float(nfalse / nsel) if nsel else 0.0

    c = class_support.astype(float)
    b = bayes_support.astype(float)
    r = rf_support.astype(float)

    def safe_corr(a: np.ndarray, d: np.ndarray) -> float:
        if len(a) < 2 or np.std(a) == 0 or np.std(d) == 0:
            return math.nan
        return float(np.corrcoef(a, d)[0, 1])

    summary["corr_classical_bayes"] = safe_corr(c, b)
    summary["corr_classical_rf"] = safe_corr(c, r)
    summary["corr_bayes_rf"] = safe_corr(b, r)
    summary["n_classical_by"] = int(class_support.sum())
    summary["n_classical_bh"] = int(class_support_bh.sum())
    summary["n_bayes"] = int(bayes_support.sum())
    summary["n_rf"] = int(rf_support.sum())
    return summary, edges


def make_cells(cfg: dict[str, Any]) -> list[Cell]:
    a = cfg["anchor"]
    r = cfg["replications"]
    cells: list[Cell] = []

    def add(family: str, scenario: str, reps: int, **changes: Any) -> None:
        p = dict(a)
        p.update(changes)
        cid_bits = [family, scenario] + [
            f"{k}{p[k]}" for k in [
                "T", "N", "K", "phi_x", "phi_y", "cross_dependence",
                "linear_effect", "nonlinear_effect",
            ]
        ]
        cid = "__".join(str(x).replace(".", "p") for x in cid_bits)
        cells.append(Cell(cell_id=cid, scenario=scenario, reps=int(reps), family=family, **p))

    for T in cfg["sweeps"]["T"]:
        add(
            "primary_global_null", "global_null", r["primary_global_null"],
            T=int(T), K=4, n_true_linear=0, n_true_nonlinear=0,
        )
        add(
            "primary_partial_null", "partial_null", r["primary_partial_null"],
            T=int(T), K=6,
            n_true_linear=min(2, int(a["n_true_linear"])),
            n_true_nonlinear=min(2, int(a["n_true_nonlinear"])),
        )

    anchor_T = int(a["T"])
    for N in cfg["sweeps"]["N"]:
        if int(N) != int(a["N"]):
            add("stress_N", "partial_null", r["stress_general"], T=anchor_T, N=int(N))
    for K in cfg["sweeps"]["K"]:
        if int(K) != int(a["K"]):
            add("stress_K", "partial_null", r["stress_high_k"], T=anchor_T, K=int(K))
    for px in cfg["sweeps"]["phi_x"]:
        if float(px) != float(a["phi_x"]):
            add("stress_phi_x", "partial_null", r["stress_general"], T=anchor_T, phi_x=float(px))
    for py in cfg["sweeps"]["phi_y"]:
        if float(py) != float(a["phi_y"]):
            add("stress_phi_y", "partial_null", r["stress_general"], T=anchor_T, phi_y=float(py))
    for cd in cfg["sweeps"]["cross_dependence"]:
        if float(cd) != float(a["cross_dependence"]):
            add("stress_crossdep", "partial_null", r["stress_general"], T=anchor_T, cross_dependence=float(cd))
    for ef in cfg["sweeps"]["linear_effect"]:
        if float(ef) != float(a["linear_effect"]):
            add(
                "power_linear", "partial_null", r["power_curve"], T=anchor_T,
                linear_effect=float(ef), nonlinear_effect=float(a["nonlinear_effect"]),
            )
    for ef in cfg["sweeps"]["nonlinear_effect"]:
        if float(ef) != float(a["nonlinear_effect"]):
            add(
                "power_nonlinear", "partial_null", r["power_curve"], T=anchor_T,
                nonlinear_effect=float(ef), linear_effect=float(a["linear_effect"]),
            )
    return cells


def _worker(payload: tuple[dict[str, Any], int, dict[str, Any], int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cell_dict, rep, cfg, base_seed = payload
    cell = Cell(**cell_dict)
    s, e = run_one_replication(cell, rep, cfg, base_seed)
    return s, e.to_dict("records")


def run_cell(cell: Cell, cfg: dict[str, Any], out_dir: Path, workers: int, chunk_size: int = 10) -> None:
    cdir = out_dir / "simulation_chunks" / cell.cell_id
    cdir.mkdir(parents=True, exist_ok=True)
    done = cdir / ".done.json"
    if done.exists() and not bool(cfg["execution"].get("overwrite_completed_stage", False)):
        log(f"SKIP completed cell {cell.cell_id}")
        return

    base_seed = int(cfg["seed"])
    for start in range(1, cell.reps + 1, chunk_size):
        end = min(cell.reps, start + chunk_size - 1)
        sf = cdir / f"summary_{start:06d}_{end:06d}.csv.gz"
        ef = cdir / f"edges_{start:06d}_{end:06d}.csv.gz"
        if sf.exists() and ef.exists():
            continue
        reps = list(range(start, end + 1))
        payloads = [(asdict(cell), rep, cfg, base_seed) for rep in reps]
        summaries: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        if workers <= 1:
            results = map(_worker, payloads)
            for s, e in results:
                summaries.append(s)
                edges.extend(e)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_worker, p) for p in payloads]
                for fut in as_completed(futures):
                    s, e = fut.result()
                    summaries.append(s)
                    edges.extend(e)
        pd.DataFrame(summaries).sort_values("rep").to_csv(sf, index=False, compression="gzip")
        pd.DataFrame(edges).sort_values(["rep", "predictor"]).to_csv(ef, index=False, compression="gzip")
        log(f"{cell.cell_id}: completed {end}/{cell.reps}")

    done.write_text(json.dumps({"cell": asdict(cell), "completed": time.time()}, indent=2), encoding="utf-8")


def aggregate_simulations(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sfiles = sorted((out_dir / "simulation_chunks").glob("*/summary_*.csv.gz"))
    efiles = sorted((out_dir / "simulation_chunks").glob("*/edges_*.csv.gz"))
    if not sfiles:
        raise RuntimeError("No simulation summary chunks were found.")
    summaries = pd.concat([pd.read_csv(f) for f in sfiles], ignore_index=True)
    edges = pd.concat([pd.read_csv(f) for f in efiles], ignore_index=True)
    summaries.to_csv(out_dir / "simulation_replications_R1.csv.gz", index=False, compression="gzip")

    group_cols = [
        "cell_id", "family", "scenario", "T", "N", "K", "phi_x", "phi_y",
        "cross_dependence", "linear_effect", "nonlinear_effect",
    ]
    agg_rows = []
    for keys, g in summaries.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        n = len(g)
        row["n_rep"] = n
        for col in [
            "submitted_triangulated_fdp", "submitted_replicated_single_fdp",
            "grouped_triangulated_fdp", "grouped_replicated_single_fdp",
        ]:
            vals = g[col].astype(float).to_numpy()
            row[f"mean_{col}"] = float(np.mean(vals))
            row[f"mcse_{col}"] = float(np.std(vals, ddof=1) / math.sqrt(n)) if n > 1 else math.nan
        for col in ["n_classical_by", "n_classical_bh", "n_bayes", "n_rf"]:
            row[f"mean_{col}"] = float(g[col].mean())
        for col in ["corr_classical_bayes", "corr_classical_rf", "corr_bayes_rf"]:
            row[f"mean_{col}"] = float(g[col].dropna().mean()) if g[col].notna().any() else math.nan
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)

    edge_agg = []
    for keys, g in edges.groupby(["cell_id", "family", "scenario"], dropna=False):
        row = {
            "cell_id": keys[0], "family": keys[1], "scenario": keys[2],
            "T": int(g["T"].iloc[0]), "N": int(g["N"].iloc[0]), "K": int(g["K"].iloc[0]),
            "phi_x": float(g["phi_x"].iloc[0]), "phi_y": float(g["phi_y"].iloc[0]),
            "cross_dependence": float(g["cross_dependence"].iloc[0]),
            "linear_effect": float(g["linear_effect"].iloc[0]),
            "nonlinear_effect": float(g["nonlinear_effect"].iloc[0]),
            "n_edge_rep": len(g),
        }
        raw_dec = g["p_lag"].astype(float) <= 0.05
        null_mask = ~g["truth"].astype(bool)
        alt_mask = g["truth"].astype(bool)
        row["classical_lag_adjusted_raw_size"] = float(raw_dec[null_mask].mean()) if int(null_mask.sum()) else math.nan
        row["classical_lag_adjusted_raw_power"] = float(raw_dec[alt_mask].mean()) if int(alt_mask.sum()) else math.nan

        for col in ["classical_by", "classical_bh", "bayes", "rf"]:
            null = g.loc[null_mask, col].astype(float)
            alt = g.loc[alt_mask, col].astype(float)
            lin_alt = g.loc[g["signal_type"].eq("linear"), col].astype(float)
            non_alt = g.loc[g["signal_type"].eq("nonlinear"), col].astype(float)
            row[f"{col}_false_positive_rate"] = float(null.mean()) if len(null) else math.nan
            row[f"{col}_power"] = float(alt.mean()) if len(alt) else math.nan
            row[f"{col}_power_linear"] = float(lin_alt.mean()) if len(lin_alt) else math.nan
            row[f"{col}_power_nonlinear"] = float(non_alt.mean()) if len(non_alt) else math.nan

        cb = g[["classical_by", "bayes", "rf"]].astype(float)
        for a_name, b_name in [
            ("classical_by", "bayes"), ("classical_by", "rf"), ("bayes", "rf")
        ]:
            aa = cb[a_name].to_numpy()
            bb = cb[b_name].to_numpy()
            key = f"{a_name}__{b_name}"
            row[f"joint_{key}"] = float(np.mean((aa > 0.5) & (bb > 0.5)))
            row[f"p_{b_name}_given_{a_name}"] = float(np.mean(bb[aa > 0.5] > 0.5)) if np.any(aa > 0.5) else math.nan
            row[f"phi_{key}"] = float(np.corrcoef(aa, bb)[0, 1]) if np.std(aa) > 0 and np.std(bb) > 0 else math.nan

        for tiercol, prefix in [("tier_submitted", "submitted"), ("tier_grouped", "grouped")]:
            for tier_name, short in [
                ("triangulated temporal precedence", "triangulated"),
                ("replicated single-family evidence", "replicated_single"),
            ]:
                sel = g[tiercol].eq(tier_name)
                null = ~g["truth"].astype(bool)
                alt = g["truth"].astype(bool)
                lin_alt = g["signal_type"].eq("linear")
                non_alt = g["signal_type"].eq("nonlinear")
                row[f"{prefix}_{short}_null_selection_rate"] = float(sel[null].mean()) if int(null.sum()) else math.nan
                row[f"{prefix}_{short}_power"] = float(sel[alt].mean()) if int(alt.sum()) else math.nan
                row[f"{prefix}_{short}_power_linear"] = float(sel[lin_alt].mean()) if int(lin_alt.sum()) else math.nan
                row[f"{prefix}_{short}_power_nonlinear"] = float(sel[non_alt].mean()) if int(non_alt.sum()) else math.nan
        edge_agg.append(row)

    edge_summary = pd.DataFrame(edge_agg)
    agg.to_csv(out_dir / "operating_characteristics_R1.csv", index=False)
    edge_summary.to_csv(out_dir / "branch_and_tier_rates_R1.csv", index=False)
    return agg, edge_summary
