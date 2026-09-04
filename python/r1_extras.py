from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from r1_core import Cell, log, pooled_linear_edge, simulate_panel, stable_seed
from r1_rf_core import rf_edge


def _null_worker(payload: tuple[dict[str, Any], int, int, dict[str, Any]]) -> list[dict[str, Any]]:
    cfg, T, rep, a = payload
    cell = Cell(
        cell_id=f"nullcal_T{T}", scenario="global_null", reps=1, T=int(T), N=int(a["N"]), K=1,
        phi_x=float(a["phi_x"]), phi_y=float(a["phi_y"]), cross_dependence=float(a["cross_dependence"]),
        linear_effect=0.0, nonlinear_effect=0.0, n_true_linear=0, n_true_nonlinear=0,
        correlated_predictor_blocks=False, family="null_calibration",
    )
    seed = stable_seed(int(cfg["seed"]), "nullcal", T, rep)
    Y, X, _ = simulate_panel(cell, seed)
    lin = pooled_linear_edge(Y, X[:, :, 0], int(cfg["max_lag"]), control="time_fe")
    circ = rf_edge(Y, X[:, :, 0], cfg, stable_seed(seed, "circ"), shadow_type="circular")
    out = []
    for block in cfg["null_calibration"]["block_lengths"]:
        blk = rf_edge(
            Y, X[:, :, 0], cfg, stable_seed(seed, "block", block),
            shadow_type="block", block=int(block),
        )
        out.append({
            "T": int(T), "rep": rep, "block_length": int(block),
            "classical_p_lag": lin["p_lag"],
            "rf_circular_supported": circ["rf_supported"],
            "rf_circular_p": circ["rf_emp_p"],
            "rf_circular_gain": circ["rf_mean"],
            "rf_circular_shadow_q95": circ["rf_shadow_q95"],
            "rf_block_supported": blk["rf_supported"],
            "rf_block_p": blk["rf_emp_p"],
            "rf_block_gain": blk["rf_mean"],
            "rf_block_shadow_q95": blk["rf_shadow_q95"],
        })
    return out


def run_null_calibration(cfg: dict[str, Any], out_dir: Path, workers: int) -> None:
    part_dir = out_dir / "null_calibration_chunks"
    part_dir.mkdir(parents=True, exist_ok=True)
    a = dict(cfg["anchor"])
    outer = int(cfg["replications"]["null_calibration_outer"])
    parts: list[Path] = []
    for T in cfg["sweeps"]["T"]:
        part = part_dir / f"null_calibration_T{int(T)}.csv"
        parts.append(part)
        if part.exists() and not bool(cfg["execution"].get("overwrite_completed_stage", False)):
            log(f"SKIP null calibration T={T} (checkpoint exists)")
            continue
        payloads = [(cfg, int(T), rep, a) for rep in range(1, outer + 1)]
        rows: list[dict[str, Any]] = []
        if workers <= 1:
            for p in payloads:
                rows.extend(_null_worker(p))
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_null_worker, p) for p in payloads]
                for j, fut in enumerate(as_completed(futures), 1):
                    rows.extend(fut.result())
                    if j % 25 == 0:
                        log(f"null calibration T={T}: {j}/{outer}")
        pd.DataFrame(rows).sort_values(["rep", "block_length"]).to_csv(part, index=False)
        log(f"null calibration T={T}: complete")
    pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(
        out_dir / "null_calibration_R1.csv", index=False
    )


def _tuning_worker(payload: tuple[dict[str, Any], int, dict[str, Any]]) -> list[dict[str, Any]]:
    cfg, rep, a = payload
    cell = Cell(
        cell_id="rf_tuning", scenario="partial_null", reps=1,
        T=int(a["T"]), N=int(a["N"]), K=max(2, int(a["K"])),
        phi_x=float(a["phi_x"]), phi_y=float(a["phi_y"]),
        cross_dependence=float(a["cross_dependence"]),
        linear_effect=float(a["linear_effect"]), nonlinear_effect=float(a["nonlinear_effect"]),
        n_true_linear=0, n_true_nonlinear=1,
        correlated_predictor_blocks=bool(a["correlated_predictor_blocks"]), family="rf_tuning",
    )
    seed = stable_seed(int(cfg["seed"]), "rf-tuning", rep)
    Y, X, truth = simulate_panel(cell, seed)
    rows: list[dict[str, Any]] = []
    for k in [0, cell.K - 1]:
        for setting in cfg["rf"]["tuning_settings"]:
            for seed_index in range(1, int(setting["seed_repeats"]) + 1):
                r = rf_edge(
                    Y, X[:, :, k], cfg,
                    stable_seed(seed, k, setting["name"], seed_index),
                    trees=int(setting["trees"]), shadow_repeats=int(setting["shadows"]),
                )
                for fold_threshold in cfg["rf"]["tuning_fold_share"]:
                    supported_at_threshold = bool(
                        np.isfinite(r["rf_mean"]) and r["rf_mean"] > 0
                        and np.isfinite(r["rf_shadow_q95"]) and r["rf_mean"] > r["rf_shadow_q95"]
                        and r["rf_fold_share"] >= float(fold_threshold)
                    )
                    rows.append({
                        "rep": rep, "predictor": k + 1, "truth": bool(truth[k]),
                        "setting": str(setting["name"]), "trees": int(setting["trees"]),
                        "shadows": int(setting["shadows"]), "seed_index": seed_index,
                        "fold_share_threshold": float(fold_threshold),
                        "supported_at_threshold": supported_at_threshold, **r,
                    })
    return rows


def run_tuning(cfg: dict[str, Any], out_dir: Path, workers: int) -> None:
    reps = int(cfg["replications"]["tuning"])
    part_dir = out_dir / "rf_tuning_chunks"
    part_dir.mkdir(parents=True, exist_ok=True)
    a = dict(cfg["anchor"])
    missing = [rep for rep in range(1, reps + 1) if not (part_dir / f"rep_{rep:04d}.csv").exists()]
    if missing:
        payloads = [(cfg, rep, a) for rep in missing]
        if workers <= 1:
            for p in payloads:
                rows = _tuning_worker(p)
                pd.DataFrame(rows).to_csv(part_dir / f"rep_{p[1]:04d}.csv", index=False)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_tuning_worker, p): p[1] for p in payloads}
                for j, fut in enumerate(as_completed(futs), 1):
                    rep = futs[fut]
                    pd.DataFrame(fut.result()).to_csv(part_dir / f"rep_{rep:04d}.csv", index=False)
                    log(f"RF tuning: {j}/{len(missing)} new replication(s)")
    files = sorted(part_dir.glob("rep_*.csv"))
    pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(
        out_dir / "rf_tuning_stability_R1.csv", index=False
    )


def generate_benchmark_inputs(cfg: dict[str, Any], out_dir: Path) -> None:
    bdir = out_dir / "benchmark_inputs"
    bdir.mkdir(parents=True, exist_ok=True)
    a = dict(cfg["anchor"])
    reps = int(cfg["replications"]["benchmark"])
    K = min(10, int(a["K"]))
    for T in cfg["sweeps"]["T"]:
        for scenario in ["global_null", "partial_null"]:
            target = bdir / f"benchmark_{scenario}_T{int(T)}.csv.gz"
            truth_target = bdir / f"benchmark_{scenario}_T{int(T)}_truth.csv"
            if target.exists() and truth_target.exists() and not bool(cfg["execution"].get("overwrite_completed_stage", False)):
                continue
            rows = []
            truth_rows = []
            cell = Cell(
                cell_id=f"benchmark_{scenario}_T{T}", scenario=scenario, reps=reps,
                T=int(T), N=int(a["N"]), K=K,
                phi_x=float(a["phi_x"]), phi_y=float(a["phi_y"]),
                cross_dependence=float(a["cross_dependence"]),
                linear_effect=float(a["linear_effect"]), nonlinear_effect=float(a["nonlinear_effect"]),
                n_true_linear=min(int(a["n_true_linear"]), K),
                n_true_nonlinear=min(int(a["n_true_nonlinear"]), max(K - int(a["n_true_linear"]), 0)),
                correlated_predictor_blocks=bool(a["correlated_predictor_blocks"]), family="benchmark",
            )
            for rep in range(1, reps + 1):
                seed = stable_seed(int(cfg["seed"]), "benchmark", scenario, T, rep)
                Y, X, truth = simulate_panel(cell, seed)
                for t in range(cell.T):
                    for i in range(cell.N):
                        row = {"rep": rep, "unit": i + 1, "time": t + 1, "y": float(Y[t, i])}
                        for k in range(K):
                            row[f"x{k+1}"] = float(X[t, i, k])
                        rows.append(row)
                for k in range(K):
                    truth_rows.append({
                        "rep": rep, "predictor": k + 1, "truth": bool(truth[k]),
                        "signal_type": (
                            "linear" if k < cell.n_true_linear and truth[k]
                            else "nonlinear" if k < cell.n_true_linear + cell.n_true_nonlinear and truth[k]
                            else "null"
                        ),
                    })
                if rep % 50 == 0:
                    log(f"benchmark input {scenario} T={T}: {rep}/{reps}")
            pd.DataFrame(rows).to_csv(target, index=False, compression="gzip")
            pd.DataFrame(truth_rows).to_csv(truth_target, index=False)


def write_guard_sensitivity(cfg: dict[str, Any], out_dir: Path) -> None:
    rows = []
    for T in cfg["sweeps"]["T"]:
        for p in range(1, int(cfg["max_lag"]) + 1):
            effective_n = int(T) - p
            residual_df = effective_n - (1 + 2 * p)
            for guard in [6, 8, 10, 12]:
                rows.append({
                    "T": int(T), "lag_order": p, "effective_time_rows": effective_n,
                    "unrestricted_residual_df_per_complete_unit": residual_df,
                    "guard": guard, "testable_complete_unit": bool(residual_df >= guard),
                })
    pd.DataFrame(rows).to_csv(out_dir / "residual_df_guard_sensitivity_R1.csv", index=False)


def write_gate_report(cfg: dict[str, Any], out_dir: Path) -> None:
    op = pd.read_csv(out_dir / "operating_characteristics_R1.csv")
    primary = op[op["family"].eq("primary_global_null")].copy()
    lines = [
        "# R1 validation-gate report", "",
        "This report is generated before any MethodsX manuscript editing.",
        "It is diagnostic: failed gates trigger a reviewer-focused method revision, not concealment.",
        "", "## Primary global-null cells", "",
    ]
    for _, r in primary.iterrows():
        lines.append(
            f"- T={int(r['T'])}, N={int(r['N'])}, K={int(r['K'])}, n={int(r['n_rep'])}: "
            f"submitted-tier empirical FDR(triangulated)={r['mean_submitted_triangulated_fdp']:.4f} "
            f"(MCSE {r['mcse_submitted_triangulated_fdp']:.4f}); "
            f"grouped-family={r['mean_grouped_triangulated_fdp']:.4f} "
            f"(MCSE {r['mcse_grouped_triangulated_fdp']:.4f})."
        )
    lines += [
        "", "## Interpretation gates", "",
        "1. Do not claim formal FDR control for the whole tier procedure solely from within-branch correction.",
        "2. Compare BH and BY results and use the reviewer-requested dependence evidence when selecting the R1 multiplicity wording.",
        "3. If classical/Bayesian null concordance is high, do not count them as independent method families in the highest tier.",
        "4. If circular-shift calibration is materially mis-sized at short T relative to the non-wrapping block surrogate, revise the null-calibration rule before manuscript editing.",
        "5. If the asymptotic dynamic-panel screen over-rejects, use its measured finite-T size and the requested external panel-Granger benchmarks rather than defending coefficient bias as irrelevant.",
        "6. The generated-regressor issue is intentionally handled by explicit limitation unless a two-stage bootstrap is separately enabled by the author.",
    ]
    (out_dir / "VALIDATION_GATES_R1.md").write_text("\n".join(lines), encoding="utf-8")


def _config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()


def write_manifest(cfg: dict[str, Any], out_dir: Path, mode: str) -> None:
    target = out_dir / "validation_manifest_R1.json"
    new_hash = _config_hash(cfg)
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        old_hash = old.get("config_sha256")
        if old_hash and old_hash != new_hash:
            raise RuntimeError(
                "Existing R1 output was created with a different configuration. "
                "Use a new --out directory rather than mixing configurations."
            )
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": mode, "python": sys.version, "platform": platform.platform(),
        "seed": cfg["seed"], "config_sha256": new_hash,
    }
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def resolve_workers(value: Any) -> int:
    if str(value).lower() == "auto":
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(value))


def smoke_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    c = json.loads(json.dumps(cfg))
    c["replications"] = {k: 2 for k in c["replications"]}
    c["anchor"].update({"T": 20, "N": 10, "K": 4, "n_true_linear": 1, "n_true_nonlinear": 1})
    c["sweeps"] = {
        "T": [20], "N": [10], "K": [4],
        "phi_x": [c["anchor"]["phi_x"]], "phi_y": [c["anchor"]["phi_y"]],
        "cross_dependence": [c["anchor"]["cross_dependence"]],
        "linear_effect": [c["anchor"]["linear_effect"]],
        "nonlinear_effect": [c["anchor"]["nonlinear_effect"]],
    }
    c["rf"]["trees_mc"] = 10
    c["rf"]["shadow_repeats_mc"] = 3
    c["rf"]["tuning_settings"] = [
        {"name": "smoke", "trees": 10, "shadows": 3, "seed_repeats": 1}
    ]
    c["tiering"]["placebo_repeats"] = 3
    c["null_calibration"]["block_lengths"] = [2]
    return c
